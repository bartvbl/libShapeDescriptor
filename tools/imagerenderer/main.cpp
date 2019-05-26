#include <spinImage/cpu/types/HostMesh.h>
#include <spinImage/utilities/OBJLoader.h>
#include <spinImage/utilities/copy/hostMeshToDevice.h>
#include <spinImage/gpu/spinImageGenerator.cuh>
#include <spinImage/gpu/quasiSpinImageGenerator.cuh>
#include <spinImage/utilities/dumpers/spinImageDumper.h>
#include <spinImage/utilities/copy/deviceDescriptorsToHost.h>
#include "arrrgh.hpp"
#include <spinImage/utilities/CUDAContextCreator.h>

int main(int argc, const char** argv) {
    arrrgh::parser parser("imagerenderer", "Generate (quasi) spin images from an input object and dump them into a PNG file");
    const auto& inputFile = parser.add<std::string>(
            "input", "The location of the input OBJ model file.", '\0', arrrgh::Required, "");
    const auto& showHelp = parser.add<bool>(
            "help", "Show this help message.", 'h', arrrgh::Optional, false);
    const auto& generationMode = parser.add<std::string>(
            "image-type", "Which image type to generate. Can either be 'spinimage' or 'quasispinimage'.", '\0', arrrgh::Optional, "quasispinimage");
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
    HostMesh mesh = SpinImage::utilities::loadOBJ(inputFile.value());
    DeviceMesh deviceMesh = SpinImage::copy::hostMeshToDevice(mesh);

    std::cout << "Generating images.. (this can take a while)" << std::endl;
    if(generationMode.value() == "spinimage") {
        array<spinImagePixelType> descriptors = SpinImage::gpu::generateSpinImages(deviceMesh, spinImageWidth.value(), supportAngle.value(), spinImageSampleCount.value());
        std::cout << "Dumping results.. " << std::endl;
        array<spinImagePixelType> hostDescriptors = SpinImage::copy::spinImageDescriptorsToHost(descriptors, deviceMesh.vertexCount);
        if(imageLimit.value() != -1) {
            hostDescriptors.length = std::min<int>(hostDescriptors.length, imageLimit.value());
        }
        SpinImage::dump::descriptors(hostDescriptors, outputFile.value(), true, 50);

        cudaFree(descriptors.content);
        delete[] hostDescriptors.content;

    } else if(generationMode.value() == "quasispinimage") {
        array<quasiSpinImagePixelType> descriptors = SpinImage::gpu::generateQuasiSpinImages(deviceMesh, spinImageWidth.value());
        std::cout << "Dumping results.. " << std::endl;
        array<quasiSpinImagePixelType> hostDescriptors = SpinImage::copy::QSIDescriptorsToHost(descriptors, deviceMesh.vertexCount);
        if(imageLimit.value() != -1) {
            hostDescriptors.length = std::min<int>(hostDescriptors.length, imageLimit.value());
        }
        SpinImage::dump::descriptors(hostDescriptors, outputFile.value(), true, 50);

        cudaFree(descriptors.content);
        delete[] hostDescriptors.content;

    } else {
        std::cerr << "Unrecognised image type: " << generationMode.value() << std::endl;
        std::cerr << "Should be either 'spinimage' or 'quasispinimage'." << std::endl;
    }

    SpinImage::cpu::freeHostMesh(mesh);
    SpinImage::gpu::freeDeviceMesh(deviceMesh);
}