#include <shapeDescriptor/shapeDescriptor.h>
#include <shapeDescriptor/descriptors/ShapeContextGenerator.h>

#include <chrono>
#include <iostream>
#include <cmath>
#include <cstring>
#include <glm/glm.hpp>



ShapeDescriptor::cpu::array<ShapeDescriptor::UniqueShapeContextDescriptor> ShapeDescriptor::generateUniqueShapeContextDescriptorsMultiRadius(
        const ShapeDescriptor::cpu::PointCloud& pointCloud,
        const ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint>& imageOrigins,
        float pointDensityRadius,
        const std::vector<float>& minSupportRadius,
        const std::vector<float>& maxSupportRadius,
        ShapeDescriptor::SCExecutionTimes* executionTimes) {
    std::chrono::time_point totalExecutionTimeStart = std::chrono::steady_clock::now();

    assert(imageOrigins.length == minSupportRadius.size());
    assert(imageOrigins.length == maxSupportRadius.size());

    ShapeDescriptor::cpu::array<ShapeDescriptor::UniqueShapeContextDescriptor> descriptors(imageOrigins.length);
    std::memset(descriptors.content, 0, descriptors.length * sizeof(UniqueShapeContextDescriptor));

    // -- Point Count Computation --
    std::chrono::time_point pointCountingStart = std::chrono::steady_clock::now();

    std::vector<uint32_t> pointCountArray = ShapeDescriptor::computePointDensities(pointDensityRadius, pointCloud);

    std::chrono::milliseconds pointCountingDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - pointCountingStart);

    // -- USC Generation --
    auto generationStart = std::chrono::steady_clock::now();

    std::vector<ShapeDescriptor::LocalReferenceFrame> referenceFrames = ShapeDescriptor::internal::computeSHOTReferenceFrames(pointCloud, imageOrigins, maxSupportRadius);

    ShapeDescriptor::computeGeneralShapeContextDescriptors<
            ShapeDescriptor::UniqueShapeContextDescriptor::horizontalSliceCount,
            ShapeDescriptor::UniqueShapeContextDescriptor::verticalSliceCount,
            ShapeDescriptor::UniqueShapeContextDescriptor::layerCount>(
            imageOrigins,
            pointCloud,
            descriptors,
            pointCountArray,
            referenceFrames,
            minSupportRadius,
            maxSupportRadius);

    std::chrono::milliseconds generationDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - generationStart);

    // -- Cleanup --

    std::chrono::milliseconds totalExecutionDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - totalExecutionTimeStart);

    if(executionTimes != nullptr) {
        executionTimes->totalExecutionTimeSeconds = double(totalExecutionDuration.count()) / 1000.0;
        executionTimes->generationTimeSeconds = double(generationDuration.count()) / 1000.0;
        executionTimes->pointCountingTimeSeconds = double(pointCountingDuration.count()) / 1000.0;
    }

    for(uint32_t i = 0; i < descriptors.length; i++) {
        for(float content : descriptors[i].contents) {
            if(std::isnan(content) || std::isinf(content)) {
                throw std::runtime_error("Found a NaN!");
            }
        }
    }

    return descriptors;
}

ShapeDescriptor::cpu::array<ShapeDescriptor::UniqueShapeContextDescriptor> ShapeDescriptor::generateUniqueShapeContextDescriptors(
        ShapeDescriptor::cpu::PointCloud pointCloud,
        ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> imageOrigins,
        float pointDensityRadius,
        float minSupportRadius,
        float maxSupportRadius,
        ShapeDescriptor::SCExecutionTimes* executionTimes) {
    std::vector<float> minRadii(imageOrigins.length, minSupportRadius);
    std::vector<float> maxRadii(imageOrigins.length, maxSupportRadius);
    return generateUniqueShapeContextDescriptorsMultiRadius(pointCloud, imageOrigins, pointDensityRadius, minRadii, maxRadii, executionTimes);
}


