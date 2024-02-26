#include <arrrgh.hpp>
#include <shapeDescriptor/shapeDescriptor.h>

int main(int argc, const char** argv) {
    const std::string defaultExecutionDevice = ShapeDescriptor::isCUDASupportAvailable() ? "gpu" : "cpu";

    arrrgh::parser parser("imagerenderer", "Generate RICI or spin images from an input object and dump them into a PNG file");
    const auto& inputFile = parser.add<std::string>(
            "input", "The location of the input OBJ model file.", '\0', arrrgh::Required, "");
    const auto& showHelp = parser.add<bool>(
            "help", "Show this help message.", 'h', arrrgh::Optional, false);
    const auto& generationMode = parser.add<std::string>(
            "image-type", "Which image type to generate. Can either be 'si', 'rici', or 'quicci'.", '\0', arrrgh::Optional, "rici");
    const auto& forceGPU = parser.add<int>(
            "force-gpu", "Force using the GPU with the given ID", 'b', arrrgh::Optional, -1);
    const auto& generationDevice = parser.add<std::string>(
            "device", "Determines whether to compute the images on the CPU or GPU, by specifying its value as 'cpu' or 'gpu', respectively.", '\0', arrrgh::Optional, defaultExecutionDevice);
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
        ShapeDescriptor::createCUDAContext(forceGPU.value());
    }

    if(!ShapeDescriptor::isCUDASupportAvailable() && generationDevice.value() == "gpu") {
        throw std::runtime_error("Image generation on the GPU was requested, but libShapeDescriptor was compiled GPU kernels disabled.");
    }

    std::cout << "Loading mesh file.." << std::endl;
    ShapeDescriptor::cpu::Mesh mesh = ShapeDescriptor::loadMesh(inputFile.value(), ShapeDescriptor::RecomputeNormals::DO_NOT_RECOMPUTE);
    std::cout << "    Object has " << mesh.vertexCount << " vertices" << std::endl;

    ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> spinOrigins = ShapeDescriptor::generateUniqueSpinOriginBuffer(mesh);
    std::cout << "    Found " << spinOrigins.length << " unique vertices" << std::endl;

    // Limit image count being generated depending on command line parameter
    if(imageLimit.value() != -1) {
        spinOrigins.length = std::min<size_t>(spinOrigins.length, imageLimit.value());
    }

    ShapeDescriptor::gpu::Mesh deviceMesh;
    ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> deviceSpinOrigins;

    if(ShapeDescriptor::isCUDASupportAvailable() && generationDevice.value() == "gpu") {
        deviceMesh = ShapeDescriptor::copyToGPU(mesh);

        ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> tempOrigins = ShapeDescriptor::copyToGPU(spinOrigins);
        deviceSpinOrigins = {tempOrigins.length, reinterpret_cast<ShapeDescriptor::OrientedPoint*>(tempOrigins.content)};
    }




    std::cout << "Generating images.. (this can take a while)" << std::endl;
    if(generationMode.value() == "si") {
        ShapeDescriptor::cpu::PointCloud pointCloud = ShapeDescriptor::sampleMesh(mesh, spinImageSampleCount.value(), 0);
        ShapeDescriptor::gpu::PointCloud device_cloud = ShapeDescriptor::copyToGPU(pointCloud);

        ShapeDescriptor::gpu::array<ShapeDescriptor::SpinImageDescriptor> descriptors = ShapeDescriptor::generateSpinImages(
                device_cloud,
                deviceSpinOrigins,
                spinImageWidth.value(),
                supportAngle.value());
        std::cout << "Dumping results.. " << std::endl;
        ShapeDescriptor::cpu::array<ShapeDescriptor::SpinImageDescriptor> hostDescriptors = ShapeDescriptor::copyToCPU<ShapeDescriptor::SpinImageDescriptor>(descriptors);
        ShapeDescriptor::writeDescriptorImages(hostDescriptors, outputFile.value(), enableLogarithmicImage.value(), imagesPerRow.value());

        ShapeDescriptor::free<ShapeDescriptor::SpinImageDescriptor>(descriptors);
        ShapeDescriptor::free(hostDescriptors);
        ShapeDescriptor::free(device_cloud);

    } else if(generationMode.value() == "rici") {
        ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor> hostDescriptors;
        if(generationDevice.value() == "gpu") {
            ShapeDescriptor::gpu::array<ShapeDescriptor::RICIDescriptor> descriptors =
                    ShapeDescriptor::generateRadialIntersectionCountImages(
                            deviceMesh,
                            deviceSpinOrigins,
                            spinImageWidth.value());
            hostDescriptors = ShapeDescriptor::copyToCPU(descriptors);
            ShapeDescriptor::free(descriptors);
        } else if(generationDevice.value() == "cpu") {
            hostDescriptors = ShapeDescriptor::generateRadialIntersectionCountImages(mesh, spinOrigins, spinImageWidth.value());
        }


        std::cout << "Dumping results.. " << std::endl;

        if(imageLimit.value() != -1) {
            hostDescriptors.length = std::min<int>(hostDescriptors.length, imageLimit.value());
        }
        ShapeDescriptor::writeDescriptorImages(hostDescriptors, outputFile.value(), enableLogarithmicImage.value(), imagesPerRow.value());
        delete[] hostDescriptors.content;



    } else if(generationMode.value() == "quicci") {
        ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> hostDescriptors;

        if(generationDevice.value() == "gpu") {
            ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> images = ShapeDescriptor::generateQUICCImages(
                    deviceMesh,
                    deviceSpinOrigins,
                    spinImageWidth.value());
            hostDescriptors = ShapeDescriptor::copyToCPU(images);
            ShapeDescriptor::free(images);
        } else if(generationDevice.value() == "cpu") {
            hostDescriptors = ShapeDescriptor::generateQUICCImages(mesh, spinOrigins, spinImageWidth.value());
        }

        std::cout << "Dumping results.. " << std::endl;

        if(imageLimit.value() != -1) {
            hostDescriptors.length = std::min<int>(hostDescriptors.length, imageLimit.value());
        }

        ShapeDescriptor::writeDescriptorImages(hostDescriptors, outputFile.value(), false, imagesPerRow.value());

        ShapeDescriptor::free(hostDescriptors);
    } else {
        std::cerr << "Unrecognised image type: " << generationMode.value() << std::endl;
        std::cerr << "Should be either 'si', 'rici', or 'quicci'." << std::endl;
    }

    ShapeDescriptor::free(mesh);
    if(generationDevice.value() == "gpu") {
        ShapeDescriptor::free(deviceMesh);
    }
}
