#include <shapeDescriptor/cpu/types/Mesh.h>
#include <shapeDescriptor/gpu/spinImageGenerator.cuh>
#include <shapeDescriptor/gpu/radialIntersectionCountImageGenerator.cuh>
#include <shapeDescriptor/gpu/quickIntersectionCountImageGenerator.cuh>
#include <shapeDescriptor/utilities/mesh/OBJLoader.h>
#include <shapeDescriptor/utilities/dumpers/descriptors.h>
#include <shapeDescriptor/utilities/CUDAContextCreator.h>
#include <shapeDescriptor/utilities/kernels/spinOriginBufferGenerator.h>

#include <arrrgh.hpp>
#include <shapeDescriptor/utilities/copy/mesh.h>
#include <shapeDescriptor/utilities/kernels/meshSampler.cuh>
#include <shapeDescriptor/utilities/copy/array.h>

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
        SpinImage::utilities::createCUDAContext(forceGPU.value());
    }

    std::cout << "Loading OBJ file.." << std::endl;
    SpinImage::cpu::Mesh mesh = SpinImage::utilities::loadOBJ(inputFile.value());
    SpinImage::gpu::Mesh deviceMesh = SpinImage::copy::hostMeshToDevice(mesh);

    SpinImage::gpu::PointCloud pointCloud = SpinImage::utilities::sampleMesh(deviceMesh, spinImageSampleCount.value(), 0);

    SpinImage::gpu::array<SpinImage::gpu::DeviceOrientedPoint> spinOrigins = SpinImage::utilities::generateUniqueSpinOriginBuffer(deviceMesh);
    size_t imageCount = spinOrigins.length;

    std::cout << "Generating images.. (this can take a while)" << std::endl;
    if(generationMode.value() == "si") {
        SpinImage::gpu::array<SpinImage::gpu::SpinImageDescriptor> descriptors = SpinImage::gpu::generateSpinImages(
                pointCloud,
                spinOrigins,
                spinImageWidth.value(),
                supportAngle.value());
        std::cout << "Dumping results.. " << std::endl;
        SpinImage::cpu::array<SpinImage::gpu::SpinImageDescriptor> hostDescriptors = SpinImage::copy::deviceArrayToHost<SpinImage::gpu::SpinImageDescriptor>(descriptors);
        if(imageLimit.value() != -1) {
            hostDescriptors.length = std::min<int>(hostDescriptors.length, imageLimit.value());
        }
        SpinImage::dump::descriptors(hostDescriptors, outputFile.value(), true, imagesPerRow.value());

        cudaFree(descriptors.content);
        delete[] hostDescriptors.content;

    } else if(generationMode.value() == "rici") {
        SpinImage::gpu::array<SpinImage::gpu::RICIDescriptor> descriptors =
            SpinImage::gpu::generateRadialIntersectionCountImages(
                deviceMesh,
                spinOrigins,
                spinImageWidth.value());

        std::cout << "Dumping results.. " << std::endl;
        SpinImage::cpu::array<SpinImage::gpu::RICIDescriptor> hostDescriptors =
                SpinImage::copy::deviceArrayToHost(descriptors);
        if(imageLimit.value() != -1) {
            hostDescriptors.length = std::min<int>(hostDescriptors.length, imageLimit.value());
        }
        SpinImage::dump::descriptors(hostDescriptors, outputFile.value(), true, imagesPerRow.value());
        delete[] hostDescriptors.content;

        cudaFree(descriptors.content);

    } else if(generationMode.value() == "quicci") {
        SpinImage::gpu::array<SpinImage::gpu::QUICCIDescriptor> images = SpinImage::gpu::generateQUICCImages(deviceMesh,
                                                                                  spinOrigins,
                                                                                  spinImageWidth.value());

        std::cout << "Dumping results.. " << std::endl;

        SpinImage::cpu::array<SpinImage::gpu::QUICCIDescriptor> host_images = SpinImage::copy::deviceArrayToHost(images);

        SpinImage::dump::descriptors(host_images, outputFile.value(), imagesPerRow.value());

        cudaFree(images.content);
        delete[] host_images.content;
    } else {
        std::cerr << "Unrecognised image type: " << generationMode.value() << std::endl;
        std::cerr << "Should be either 'si', 'rici', or 'quicci'." << std::endl;
    }

    SpinImage::cpu::freeMesh(mesh);
    SpinImage::gpu::freeMesh(deviceMesh);

}