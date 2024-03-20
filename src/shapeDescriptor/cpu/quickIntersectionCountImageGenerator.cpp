#include <chrono>
#include <cstring>
#include <shapeDescriptor/shapeDescriptor.h>
#include <bitset>

void generateQUICCIDescriptor(const ShapeDescriptor::RICIDescriptor &riciDescriptor, ShapeDescriptor::QUICCIDescriptor* descriptor, const int minPixelDelta) {
    static_assert(sizeof(unsigned int) == 4, "The logic of this function assumes 32-bit integers, which differs from your system. You will need to modify this function to make it work properly.");
    static_assert(spinImageWidthPixels % 32 == 0, "The size of a QUICCI descriptor must be a multiple of 32");

    unsigned int nextChunk = 0;
    for(int row = 0; row < spinImageWidthPixels; row++) {
        // Since we compute deltas, we start from column 1 to avoid out of range errors
        for(int column = 1; column < spinImageWidthPixels; column++) {
            int currentPixelValue = int(riciDescriptor.contents[spinImageWidthPixels * row + column]);
            int previousPixelValue = int(riciDescriptor.contents[spinImageWidthPixels * row + column - 1]);
            int pixelDelta = currentPixelValue - previousPixelValue;
            // Check if the change in intersection count is more than 1
            bool pixelValue = (pixelDelta * pixelDelta) >= (minPixelDelta * minPixelDelta);

            nextChunk = nextChunk | ((pixelValue ? 1u : 0u) << (31u - unsigned(column)));

            if(column % 32 == 31) {
                descriptor->contents[(spinImageWidthPixels / 32) * row + (column / 32)] = nextChunk;
                nextChunk = 0;
            }
        }

    }
}

template<unsigned int threshold>
ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> generateQUICCImagesCPU(
        const ShapeDescriptor::cpu::Mesh mesh,
        const ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> descriptorOrigins,
        const std::vector<float>& supportRadii,
        ShapeDescriptor::QUICCIExecutionTimes* executionTimes) {
    auto totalExecutionTimeStart = std::chrono::steady_clock::now();

    assert(supportRadii.size() == descriptorOrigins.length);

    auto generationStart = std::chrono::steady_clock::now();
    ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor> riciDescriptors
            = ShapeDescriptor::generateRadialIntersectionCountImagesMultiRadius(mesh, descriptorOrigins, supportRadii);

    size_t imageCount = descriptorOrigins.length;
    ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors(imageCount);

    // -- Descriptor Generation --

#pragma omp parallel for
    for(size_t imageIndex = 0; imageIndex < descriptors.length; imageIndex++) {
        generateQUICCIDescriptor(riciDescriptors.content[imageIndex], &descriptors.content[imageIndex], threshold);
    }

    std::chrono::milliseconds generationDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - generationStart);
    std::chrono::milliseconds totalExecutionDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - totalExecutionTimeStart);

    if(executionTimes != nullptr) {
        executionTimes->totalExecutionTimeSeconds = double(totalExecutionDuration.count()) / 1000.0;
        executionTimes->generationTimeSeconds = double(generationDuration.count()) / 1000.0;
    }

    ShapeDescriptor::free(riciDescriptors);

    return descriptors;
}

ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> ShapeDescriptor::generatePartialityResistantQUICCImagesMultiRadius(
        const cpu::Mesh& mesh,
        const cpu::array<OrientedPoint>& descriptorOrigins,
        const std::vector<float>& supportRadii,
        ShapeDescriptor::QUICCIExecutionTimes* executionTimes) {
    return generateQUICCImagesCPU<2>(mesh, descriptorOrigins, supportRadii, executionTimes);
}

ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> ShapeDescriptor::generatePartialityResistantQUICCImages(
        ShapeDescriptor::cpu::Mesh mesh,
        ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> descriptorOrigins,
        float spinImageWidth,
        ShapeDescriptor::QUICCIExecutionTimes* executionTimes) {
    std::vector<float> radii(descriptorOrigins.length, spinImageWidth);
    return generateQUICCImagesCPU<2>(mesh, descriptorOrigins, radii, executionTimes);
}

ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> ShapeDescriptor::generateQUICCImages(
        ShapeDescriptor::cpu::Mesh mesh,
        ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> descriptorOrigins,
        float spinImageWidth,
        ShapeDescriptor::QUICCIExecutionTimes* executionTimes) {
    std::vector<float> radii(descriptorOrigins.length, spinImageWidth);
    return generateQUICCImagesCPU<1>(mesh, descriptorOrigins, radii, executionTimes);
}

