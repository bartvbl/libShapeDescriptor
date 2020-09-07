#include <arrrgh.hpp>
#include <shapeDescriptor/cpu/types/Mesh.h>
#include <shapeDescriptor/utilities/mesh/OBJLoader.h>
#include <shapeDescriptor/gpu/types/Mesh.h>
#include <shapeDescriptor/common/OrientedPoint.h>
#include <shapeDescriptor/gpu/quickIntersectionCountImageGenerator.cuh>
#include <shapeDescriptor/utilities/kernels/duplicateRemoval.cuh>
#include <cuda_runtime.h>
#include <shapeDescriptor/utilities/mesh/MeshScaler.h>
#include <shapeDescriptor/utilities/dumpers/rawDescriptorDumper.h>
#include <shapeDescriptor/utilities/copy/mesh.h>
#include <shapeDescriptor/utilities/copy/array.h>

const float DEFAULT_SPIN_IMAGE_WIDTH = 0.3;

int main(int argc, const char** argv) {
    arrrgh::parser parser("quiccidumper", "Render QUICCI images from an input OBJ file, and dump them to a compressed (ZIP) file in binary form.");
    const auto& inputOBJFile = parser.add<std::string>("input-obj-file", "Location of the OBJ file from which the images should be rendered.", '\0', arrrgh::Required, "");
    const auto& outputDumpFile = parser.add<std::string>("output-dump-file", "Location where the generated images should be dumped to.", '\0', arrrgh::Required, "");
    const auto& fitInUnitSphere = parser.add<bool>("fit-object-in-unit-sphere", "Scale the object such that it fits in a unit sphere", '\0', arrrgh::Optional, false);
    const auto& spinImageWidth = parser.add<float>("spin-image-width", "The size of the spin image plane in 3D object space", '\0', arrrgh::Optional, DEFAULT_SPIN_IMAGE_WIDTH);
    const auto& showHelp = parser.add<bool>("help", "Show this help message.", 'h', arrrgh::Optional, false);

    try
    {
        parser.parse(argc, argv);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error parsing arguments: " << e.what() << std::endl;
        parser.show_usage(std::cerr);
        exit(1);
    }

    // Show help if desired
    if(showHelp.value())
    {
        return 0;
    }

    std::cout << "Loading OBJ file: " << inputOBJFile.value() << std::endl;
    ShapeDescriptor::cpu::Mesh hostMesh = ShapeDescriptor::utilities::loadOBJ(inputOBJFile.value(), true);

    if(fitInUnitSphere.value()) {
        std::cout << "Fitting object in unit sphere.." << std::endl;
        ShapeDescriptor::cpu::Mesh scaledMesh = ShapeDescriptor::utilities::fitMeshInsideSphereOfRadius(hostMesh, 1);
        ShapeDescriptor::cpu::freeMesh(hostMesh);
        hostMesh = scaledMesh;
    }

    ShapeDescriptor::gpu::Mesh deviceMesh = ShapeDescriptor::copy::hostMeshToDevice(hostMesh);
    ShapeDescriptor::cpu::freeMesh(hostMesh);

    std::cout << "Computing QUICCI images.." << std::endl;
    ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> uniqueVertices =
            ShapeDescriptor::utilities::computeUniqueVertices(deviceMesh);

    ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> images = ShapeDescriptor::gpu::generateQUICCImages(deviceMesh, uniqueVertices, spinImageWidth.value());
    ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> hostImages = ShapeDescriptor::copy::deviceArrayToHost(images);

    std::cout << "Writing output file.." << std::endl,
            ShapeDescriptor::dump::raw::descriptors(outputDumpFile.value(), hostImages);

    ShapeDescriptor::gpu::freeMesh(deviceMesh);
    cudaFree(uniqueVertices.content);
    cudaFree(images.content);

}

