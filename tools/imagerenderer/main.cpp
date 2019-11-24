#include <spinImage/cpu/types/Mesh.h>
#include <spinImage/cpu/types/QUICCIImages.h>
#include <spinImage/gpu/spinImageGenerator.cuh>
#include <spinImage/gpu/radialIntersectionCountImageGenerator.cuh>
#include <spinImage/gpu/quickIntersectionCountImageGenerator.cuh>
#include <spinImage/utilities/OBJLoader.h>
#include <spinImage/utilities/copy/hostMeshToDevice.h>
#include <spinImage/utilities/dumpers/spinImageDumper.h>
#include <spinImage/utilities/copy/deviceDescriptorsToHost.h>
#include <spinImage/utilities/CUDAContextCreator.h>
#include <spinImage/utilities/spinOriginBufferGenerator.h>

#include <arrrgh.hpp>

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
            "spin-image-width", "The size of the spin image plane in 3D object space", '\0', arrrgh::Optional, 1.0f);
    const auto& imageLimit = parser.add<int>(
            "image-limit", "The maximum number of images to generate (in order to limit image size)", '\0', arrrgh::Optional, -1);
    const auto& supportAngle = parser.add<float>(
            "spin-image-support-angle", "The support angle to use for spin image generation", '\0', arrrgh::Optional, 90.0f);
    const auto& spinImageSampleCount = parser.add<int>(
            "spin-image-sample-count", "The number of uniformly sampled points to use for spin image generation", '\0', arrrgh::Optional, 1000000);
    const auto& imagesPerRow = parser.add<int>(
            "images-per-row", "The number of images the output image should contain per row", '\0', arrrgh::Optional, 50);
    const auto& outputFile = parser.add<std::string>(
            "output", "The maximum number of images to generate (in order to limit image size)", '\0', arrrgh::Optional, "out.png");

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

    SpinImage::array<SpinImage::gpu::DeviceOrientedPoint> spinOrigins = SpinImage::utilities::generateUniqueSpinOriginBuffer(deviceMesh);
    size_t imageCount = spinOrigins.length;

    std::cout << "Generating images.. (this can take a while)" << std::endl;
    if(generationMode.value() == "si") {
        SpinImage::array<spinImagePixelType> descriptors = SpinImage::gpu::generateSpinImages(
                deviceMesh,
                spinOrigins,
                spinImageWidth.value(),
                spinImageSampleCount.value(),
                supportAngle.value());
        std::cout << "Dumping results.. " << std::endl;
        SpinImage::array<spinImagePixelType> hostDescriptors = SpinImage::copy::spinImageDescriptorsToHost(descriptors, imageCount);
        if(imageLimit.value() != -1) {
            hostDescriptors.length = std::min<int>(hostDescriptors.length, imageLimit.value());
        }
        SpinImage::dump::descriptors(hostDescriptors, outputFile.value(), true, imagesPerRow.value());

        cudaFree(descriptors.content);
        delete[] hostDescriptors.content;

    } else if(generationMode.value() == "rici" || generationMode.value() == "quicci")  {
        SpinImage::array<radialIntersectionCountImagePixelType> descriptors = SpinImage::gpu::generateRadialIntersectionCountImages(
                deviceMesh,
                spinOrigins,
                spinImageWidth.value());
        if(generationMode.value() == "rici") {
            std::cout << "Dumping results.. " << std::endl;
            SpinImage::array<radialIntersectionCountImagePixelType> hostDescriptors = SpinImage::copy::RICIDescriptorsToHost(descriptors, imageCount);
            if(imageLimit.value() != -1) {
                hostDescriptors.length = std::min<int>(hostDescriptors.length, imageLimit.value());
            }
            SpinImage::dump::descriptors(hostDescriptors, outputFile.value(), true, imagesPerRow.value());
            delete[] hostDescriptors.content;
        } else {
            SpinImage::gpu::QUICCIImages images = SpinImage::gpu::generateQUICCImages(descriptors);

            SpinImage::cpu::QUICCIImages host_images = SpinImage::copy::QUICCIDescriptorsToHost(images);

            SpinImage::dump::descriptors(host_images, outputFile.value(), imagesPerRow.value());

            cudaFree(images.horizontallyIncreasingImages);
            cudaFree(images.horizontallyDecreasingImages);

        }
        cudaFree(descriptors.content);

    } else {
        std::cerr << "Unrecognised image type: " << generationMode.value() << std::endl;
        std::cerr << "Should be either 'si', 'rici', or 'quicci'." << std::endl;
    }

    SpinImage::cpu::freeMesh(mesh);
    SpinImage::gpu::freeMesh(deviceMesh);

}