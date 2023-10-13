#include <arrrgh.hpp>
#include <shapeDescriptor/shapeDescriptor.h>
#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
#include <cuda_runtime.h>
#endif

const float DEFAULT_SPIN_IMAGE_WIDTH = 0.3;

int main(int argc, const char** argv) {
    arrrgh::parser parser("quiccidumper", "Render QUICCI images from an input OBJ/OFF/PLY file, and dump them to a compressed (LZMA2) file in binary form.");
    const auto& inputOBJFile = parser.add<std::string>("input-obj-file", "Location of the OBJ/OFF/PLY file from which the images should be rendered.", '\0', arrrgh::Required, "");
    const auto& outputDumpFile = parser.add<std::string>("output-dump-file", "Location where the generated images should be dumped to.", '\0', arrrgh::Required, "");
    const auto& fitInUnitSphere = parser.add<bool>("fit-object-in-unit-sphere", "Scale the object such that it fits in a unit sphere", '\0', arrrgh::Optional, false);
    const auto& spinImageWidth = parser.add<float>("support-radius", "The size of the spin image plane in 3D object space", '\0', arrrgh::Optional, DEFAULT_SPIN_IMAGE_WIDTH);
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

    std::cout << "Loading mesh file: " << inputOBJFile.value() << std::endl;
    ShapeDescriptor::cpu::Mesh hostMesh = ShapeDescriptor::loadMesh(inputOBJFile.value(), ShapeDescriptor::RecomputeNormals::ALWAYS_RECOMPUTE);

    if(fitInUnitSphere.value()) {
        std::cout << "Fitting object in unit sphere.." << std::endl;
        ShapeDescriptor::cpu::Mesh scaledMesh = ShapeDescriptor::fitMeshInsideSphereOfRadius(hostMesh, 1);
        ShapeDescriptor::free(hostMesh);
        hostMesh = scaledMesh;
    }

    ShapeDescriptor::gpu::Mesh deviceMesh = ShapeDescriptor::copyToGPU(hostMesh);
    ShapeDescriptor::free(hostMesh);

    std::cout << "Computing QUICCI images.." << std::endl;
    ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> hostUniqueVertices =
            ShapeDescriptor::generateUniqueSpinOriginBuffer(hostMesh);
    ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> uniqueVertices = ShapeDescriptor::copyToGPU(hostUniqueVertices);
    ShapeDescriptor::free(hostUniqueVertices);

    ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> images = ShapeDescriptor::generateQUICCImages(deviceMesh, uniqueVertices, spinImageWidth.value());
    ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> hostImages = ShapeDescriptor::copyToCPU(images);

    std::cout << "Writing output file.." << std::endl,
            ShapeDescriptor::writeCompressedQUICCIDescriptors(outputDumpFile.value(), hostImages, 0);

    ShapeDescriptor::free(deviceMesh);
#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
    cudaFree(uniqueVertices.content);
    cudaFree(images.content);
#endif
}

