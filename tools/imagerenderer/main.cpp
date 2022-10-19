#include <shapeDescriptor/cpu/types/Mesh.h>
#include <shapeDescriptor/gpu/spinImageGenerator.cuh>
#include <shapeDescriptor/gpu/radialIntersectionCountImageGenerator.cuh>
#include <shapeDescriptor/gpu/quickIntersectionCountImageGenerator.cuh>
#include <shapeDescriptor/utilities/read/OBJLoader.h>
#include <shapeDescriptor/utilities/dump/descriptorImages.h>
#include <shapeDescriptor/utilities/CUDAContextCreator.h>
#include <shapeDescriptor/utilities/kernels/spinOriginBufferGenerator.h>
#include <shapeDescriptor/utilities/free/mesh.h>

#include <arrrgh.hpp>
#include <shapeDescriptor/utilities/copy/mesh.h>
#include <shapeDescriptor/utilities/kernels/meshSampler.cuh>
#include <shapeDescriptor/utilities/copy/array.h>
#include <shapeDescriptor/utilities/read/MeshLoader.h>
#include <shapeDescriptor/utilities/free/array.h>

int main(int argc, const char** argv) {
    arrrgh::parser parser("imagerenderer", "Generate RICI or spin images from an input object and dump them into a PNG file");
    const auto& inputFile = parser.add<std::string>(
            "input", "The location of the input OBJ model file.", '\0', arrrgh::Required, "");
    const auto& showHelp = parser.add<bool>(
            "help", "Show this help message.", 'h', arrrgh::Optional, false);
    const auto& generationMode = parser.add<std::string>(
            "image-type", "Which image type to generate. Can either be 'si', 'rici', or 'quicci'.", '\0', arrrgh::Optional, "rici");
    const auto& forceGPU = parser.add<int>(
            "force-gpu", "Force using the GPU with the given ID", 'b', arrrgh::Optional, -1);
    const auto& spinImageWidth = parser.add<float>(
            "support-radius", "The size of the spin image plane in 3D object space", '\0', arrrgh::Optional, 1.0f);
    const auto& imageLimit = parser.add<int>(
            "image-limit", "The maximum number of images to generate (in order to limit image size)", '\0', arrrgh::Optional, -1);
    const auto& enableLogarithmicImage = parser.add<bool>(
            "logarithmic-image", "Apply a logarithmic filter on the image to better show colour variation.", 'l', arrrgh::Optional, false);
    const auto& supportAngle = parser.add<float>(
            "spin-image-support-angle", "The support angle to use for spin image generation", '\0', arrrgh::Optional, 90.0f);
    const auto& spinImageSampleCount = parser.add<int>(
            "spin-image-sample-count", "The number of uniformly sampled points to use for spin image generation", '\0', arrrgh::Optional, 1000000);
    const auto& imagesPerRow = parser.add<int>(
            "images-per-row", "The number of images the output image should contain per row", '\0', arrrgh::Optional, 50);
    const auto& outputFile = parser.add<std::string>(
            "output", "The location of the PNG file to write to", '\0', arrrgh::Optional, "out.png");

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

    if(forceGPU.value() != -1) {
        std::cout << "Forcing GPU " << forceGPU.value() << std::endl;
        ShapeDescriptor::utilities::createCUDAContext(forceGPU.value());
    }

    std::cout << "Loading mesh file.." << std::endl;
    ShapeDescriptor::cpu::Mesh mesh = ShapeDescriptor::utilities::loadMesh(inputFile.value(), true);
    ShapeDescriptor::gpu::Mesh deviceMesh = ShapeDescriptor::copy::hostMeshToDevice(mesh);
    std::cout << "    Object has " << mesh.vertexCount << " vertices" << std::endl;

    std::cout << "Locating unique vertices.." << std::endl;

    size_t backupSize = deviceMesh.vertexCount;
    if(imageLimit.value() != -1) {
        deviceMesh.vertexCount = 10*imageLimit.value();
    }

    ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> spinOrigins = ShapeDescriptor::utilities::generateUniqueSpinOriginBuffer(deviceMesh);

    if(imageLimit.value() != -1) {
        deviceMesh.vertexCount = backupSize;
        spinOrigins.length = std::min<int>(spinOrigins.length, imageLimit.value());
    }

    std::cout << "Generating images.. (this can take a while)" << std::endl;
    if(generationMode.value() == "si") {
        ShapeDescriptor::gpu::PointCloud pointCloud = ShapeDescriptor::utilities::sampleMesh(deviceMesh, spinImageSampleCount.value(), 0);

        ShapeDescriptor::gpu::array<ShapeDescriptor::SpinImageDescriptor> descriptors = ShapeDescriptor::gpu::generateSpinImages(
                pointCloud,
                spinOrigins,
                spinImageWidth.value(),
                supportAngle.value());
        std::cout << "Dumping results.. " << std::endl;
        ShapeDescriptor::cpu::array<ShapeDescriptor::SpinImageDescriptor> hostDescriptors = ShapeDescriptor::copy::deviceArrayToHost<ShapeDescriptor::SpinImageDescriptor>(descriptors);
        if(imageLimit.value() != -1) {
            hostDescriptors.length = std::min<int>(hostDescriptors.length, imageLimit.value());
        }
        ShapeDescriptor::dump::descriptors(hostDescriptors, outputFile.value(), enableLogarithmicImage.value(), imagesPerRow.value());

        ShapeDescriptor::free::array<ShapeDescriptor::SpinImageDescriptor>(descriptors);
        delete[] hostDescriptors.content;

    } else if(generationMode.value() == "rici") {
        ShapeDescriptor::gpu::array<ShapeDescriptor::RICIDescriptor> descriptors =
                ShapeDescriptor::gpu::generateRadialIntersectionCountImages(
                deviceMesh,
                spinOrigins,
                spinImageWidth.value());

        std::cout << "Dumping results.. " << std::endl;
        ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor> hostDescriptors =
                ShapeDescriptor::copy::deviceArrayToHost(descriptors);
        if(imageLimit.value() != -1) {
            hostDescriptors.length = std::min<int>(hostDescriptors.length, imageLimit.value());
        }
        ShapeDescriptor::dump::descriptors(hostDescriptors, outputFile.value(), enableLogarithmicImage.value(), imagesPerRow.value());
        delete[] hostDescriptors.content;

        ShapeDescriptor::free::array(descriptors);

    } else if(generationMode.value() == "quicci") {
        ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> images = ShapeDescriptor::gpu::generateQUICCImages(deviceMesh,
                                                                                  spinOrigins,
                                                                                  spinImageWidth.value());

        std::cout << "Dumping results.. " << std::endl;

        ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> host_images = ShapeDescriptor::copy::deviceArrayToHost(images);

        if(imageLimit.value() != -1) {
            host_images.length = std::min<int>(host_images.length, imageLimit.value());
        }

        ShapeDescriptor::dump::descriptors(host_images, outputFile.value(), imagesPerRow.value());

        ShapeDescriptor::free::array(images);
        delete[] host_images.content;
    } else {
        std::cerr << "Unrecognised image type: " << generationMode.value() << std::endl;
        std::cerr << "Should be either 'si', 'rici', or 'quicci'." << std::endl;
    }

    ShapeDescriptor::free::mesh(mesh);
    ShapeDescriptor::gpu::freeMesh(deviceMesh);

}