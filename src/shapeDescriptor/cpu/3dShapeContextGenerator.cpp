#include <shapeDescriptor/shapeDescriptor.h>
#include <shapeDescriptor/descriptors/ShapeContextGenerator.h>

#include <chrono>
#include <iostream>
#include <cmath>
#include <cstring>


ShapeDescriptor::cpu::array<ShapeDescriptor::ShapeContextDescriptor> ShapeDescriptor::generate3DSCDescriptorsMultiRadius(
        const ShapeDescriptor::cpu::PointCloud& pointCloud,
        const ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint>& imageOrigins,
        float pointDensityRadius,
        const std::vector<float>& minSupportRadius,
        const std::vector<float>& maxSupportRadius,
        ShapeDescriptor::SCExecutionTimes* executionTimes) {
    std::chrono::time_point totalExecutionTimeStart = std::chrono::steady_clock::now();

    assert(imageOrigins.length == minSupportRadius.size());
    assert(imageOrigins.length == maxSupportRadius.size());

    ShapeDescriptor::cpu::array<ShapeDescriptor::ShapeContextDescriptor> descriptors(imageOrigins.length);
    std::memset(descriptors.content, 0, descriptors.length * sizeof(ShapeContextDescriptor::contents));

    // -- Point Count Computation --
    std::chrono::time_point pointCountingStart = std::chrono::steady_clock::now();

    std::vector<uint32_t> pointCountArray = ShapeDescriptor::computePointDensities(pointDensityRadius, pointCloud);

    std::chrono::milliseconds pointCountingDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - pointCountingStart);

    // -- 3DSC Generation --
    auto generationStart = std::chrono::steady_clock::now();

    std::vector<ShapeDescriptor::LocalReferenceFrame> referenceFrames(imageOrigins.length);

    for(uint32_t i = 0; i < imageOrigins.length; i++) {
        // First, we align the input vertex with the descriptor's coordinate system
        ShapeDescriptor::cpu::float3 normal = imageOrigins.content[i].normal;
        ShapeDescriptor::cpu::float3 arbitraryAxis = {0, 0, 1};
        if (normal == arbitraryAxis || -1 * normal == arbitraryAxis) {
            arbitraryAxis = {1, 0, 0};
        }

        ShapeDescriptor::LocalReferenceFrame& referenceFrame = referenceFrames.at(i);

        referenceFrame.xAxis = cross(arbitraryAxis, normal);
        referenceFrame.yAxis = cross(referenceFrame.xAxis, normal);
        referenceFrame.zAxis = normal;


        assert(length(referenceFrame.xAxis) != 0);
        assert(length(referenceFrame.yAxis) != 0);

        referenceFrame.xAxis = normalize(referenceFrame.xAxis);
        referenceFrame.yAxis = normalize(referenceFrame.yAxis);
    }

    ShapeDescriptor::computeGeneralShapeContextDescriptors<
            ShapeDescriptor::ShapeContextDescriptor::horizontalSliceCount,
            ShapeDescriptor::ShapeContextDescriptor::verticalSliceCount,
            ShapeDescriptor::ShapeContextDescriptor::layerCount>(
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

ShapeDescriptor::cpu::array<ShapeDescriptor::ShapeContextDescriptor> ShapeDescriptor::generate3DSCDescriptors(
        ShapeDescriptor::cpu::PointCloud pointCloud,
        ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> imageOrigins,
        float pointDensityRadius,
        float minSupportRadius,
        float maxSupportRadius,
        ShapeDescriptor::SCExecutionTimes* executionTimes) {
    std::vector<float> minRadii(imageOrigins.length, minSupportRadius);
    std::vector<float> maxRadii(imageOrigins.length, maxSupportRadius);
    return generate3DSCDescriptorsMultiRadius(pointCloud, imageOrigins, pointDensityRadius, minRadii, maxRadii, executionTimes);
}


