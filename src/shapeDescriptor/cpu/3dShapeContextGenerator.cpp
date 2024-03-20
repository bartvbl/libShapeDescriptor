#include <shapeDescriptor/shapeDescriptor.h>

#include <shapeDescriptor/shapeDescriptor.h>

#include <chrono>
#include <iostream>
#include <cmath>
#include <cstring>

const size_t elementsPerShapeContextDescriptor =
        SHAPE_CONTEXT_HORIZONTAL_SLICE_COUNT *
        SHAPE_CONTEXT_VERTICAL_SLICE_COUNT *
        SHAPE_CONTEXT_LAYER_COUNT;

float computeLayerDistance(float minSupportRadius, float maxSupportRadius, short layerIndex) {
    // Avoiding zero divisions
    if(minSupportRadius == 0) {
        minSupportRadius = 0.000001f;
    }
    return std::exp(
            (std::log(minSupportRadius))
            + ((float(layerIndex) / float(SHAPE_CONTEXT_LAYER_COUNT))
            * std::log(float(maxSupportRadius) / float(minSupportRadius))));
}


inline float computeWedgeSegmentVolume(short verticalBinIndex, float radius) {
    const float verticalAngleStep = 1.0f / float(SHAPE_CONTEXT_VERTICAL_SLICE_COUNT);
    float binStartAngle = float(verticalBinIndex) * verticalAngleStep;
    float binEndAngle = float(verticalBinIndex + 1) * verticalAngleStep;

    float scaleFraction = (2.0f * float(M_PI) * radius * radius * radius)
                        / (3.0f * float(SHAPE_CONTEXT_HORIZONTAL_SLICE_COUNT));
    return scaleFraction * (std::cos(binStartAngle) - std::cos(binEndAngle));
}

inline float computeSingleBinVolume(short verticalBinIndex, short layerIndex, float minSupportRadius, float maxSupportRadius) {
    // The wedge segment computation goes all the way from the center to the edge of the sphere
    // Since we also have a minimum support radius, we need to cut out the volume of the centre part
    float binEndRadius = computeLayerDistance(minSupportRadius, maxSupportRadius, layerIndex + 1);
    float binStartRadius = computeLayerDistance(minSupportRadius, maxSupportRadius, layerIndex);

    float largeSupportRadiusVolume = computeWedgeSegmentVolume(verticalBinIndex, binEndRadius);
    float smallSupportRadiusVolume = computeWedgeSegmentVolume(verticalBinIndex, binStartRadius);

    return largeSupportRadiusVolume - smallSupportRadiusVolume;
}

// Cuda is being dumb. Need to create separate function to allow the linker to figure out that, yes, this function does
// indeed exist somewhere.

float ShapeDescriptor::internal::computeBinVolume(short verticalBinIndex, short layerIndex, float minSupportRadius, float maxSupportRadius) {
    return computeSingleBinVolume(verticalBinIndex, layerIndex, minSupportRadius, maxSupportRadius);
}

float absoluteAngle(float y, float x);

// Run once for every vertex index
void createDescriptors(
    ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> spinImageOrigins,
    ShapeDescriptor::cpu::PointCloud pointCloud,
    ShapeDescriptor::cpu::array<ShapeDescriptor::ShapeContextDescriptor> descriptors,
    ShapeDescriptor::cpu::array<unsigned int> pointDensityArray,
    size_t sampleCount,
    const std::vector<float>& minSupportRadius,
    const std::vector<float>& maxSupportRadius)
{
    for(uint32_t descriptorIndex = 0; descriptorIndex < descriptors.length; descriptorIndex++) {

        const ShapeDescriptor::OrientedPoint spinOrigin = spinImageOrigins[descriptorIndex];

        const ShapeDescriptor::cpu::float3 vertex = spinOrigin.vertex;
        ShapeDescriptor::cpu::float3 normal = spinOrigin.normal;

        normal = normalize(normal);

        for(float & content : descriptors.content[descriptorIndex].contents) {
            content = 0;
        }

        // First, we align the input vertex with the descriptor's coordinate system
        ShapeDescriptor::cpu::float3 arbitraryAxis = {0, 0, 1};
        if (normal == arbitraryAxis || -1 * normal == arbitraryAxis) {
            arbitraryAxis = {1, 0, 0};
        }

        ShapeDescriptor::cpu::float3 referenceXAxis = cross(arbitraryAxis, normal);
        ShapeDescriptor::cpu::float3 referenceYAxis = cross(referenceXAxis, normal);

        assert(length(referenceXAxis) != 0);
        assert(length(referenceYAxis) != 0);

        referenceXAxis = normalize(referenceXAxis);
        referenceYAxis = normalize(referenceYAxis);

        for (unsigned int sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++) {
            // 0. Fetch sample vertex

            const ShapeDescriptor::cpu::float3 samplePoint = pointCloud.vertices[sampleIndex];

            // 1. Compute bin indices

            const ShapeDescriptor::cpu::float3 translated = samplePoint - vertex;

            // Only include vertices which are within the support radius
            float distanceToVertex = length(translated);
            if (distanceToVertex < minSupportRadius.at(descriptorIndex) || distanceToVertex > maxSupportRadius.at(descriptorIndex)) {
                continue;
            }

            // Transforming descriptor coordinate system to the origin
            // In the new system, 'z' is 'up'
            const ShapeDescriptor::cpu::float3 relativeSamplePoint = {
                    referenceXAxis.x * translated.x + referenceXAxis.y * translated.y + referenceXAxis.z * translated.z,
                    referenceYAxis.x * translated.x + referenceYAxis.y * translated.y + referenceYAxis.z * translated.z,
                    normal.x * translated.x + normal.y * translated.y + normal.z * translated.z,
            };

            ShapeDescriptor::cpu::float2 horizontalDirection = {relativeSamplePoint.x, relativeSamplePoint.y};
            ShapeDescriptor::cpu::float2 verticalDirection = {length(horizontalDirection), relativeSamplePoint.z};

            if (horizontalDirection == ShapeDescriptor::cpu::float2(0, 0)) {
                // special case, will result in an angle of 0
                horizontalDirection = {1, 0};

                // Vertical direction is only 0 if all components are 0
                // Should theoretically never occur, but let's handle it just in case
                if (verticalDirection.y == 0) {
                    verticalDirection = {1, 0};
                }
            }

            // normalise direction vectors
            horizontalDirection = normalize(horizontalDirection);
            verticalDirection = normalize(verticalDirection);

            float horizontalAngle = absoluteAngle(horizontalDirection.y, horizontalDirection.x);
            short horizontalIndex =
                    unsigned((horizontalAngle / (2.0f * float(M_PI))) *
                             float(SHAPE_CONTEXT_HORIZONTAL_SLICE_COUNT))
                    % SHAPE_CONTEXT_HORIZONTAL_SLICE_COUNT;

            float verticalAngle = std::fmod(
                    absoluteAngle(verticalDirection.y, verticalDirection.x) + (float(M_PI) / 2.0f),
                    2.0f * float(M_PI));
            short verticalIndex =
                    unsigned((verticalAngle / M_PI) *
                             float(SHAPE_CONTEXT_VERTICAL_SLICE_COUNT))
                    % SHAPE_CONTEXT_VERTICAL_SLICE_COUNT;

            float sampleDistance = length(relativeSamplePoint);
            short layerIndex = 0;

            // Recomputing logarithms is still preferable over doing memory transactions for each of them
            for (; layerIndex < SHAPE_CONTEXT_LAYER_COUNT; layerIndex++) {
                float nextSliceEnd = computeLayerDistance(minSupportRadius.at(descriptorIndex), maxSupportRadius.at(descriptorIndex), layerIndex + 1);
                if (sampleDistance <= nextSliceEnd) {
                    break;
                }
            }

            // Rounding errors can cause it to exceed its allowed bounds in specific cases
            // Of course, on the off chance something is wrong after all,
            // the assertions further down should trip. So we only handle the single
            // edge case where layerIndex went one over.
            if (layerIndex == SHAPE_CONTEXT_LAYER_COUNT) {
                layerIndex--;
            }

            short3 binIndex = {horizontalIndex, verticalIndex, layerIndex};
            assert(binIndex.x >= 0);
            assert(binIndex.y >= 0);
            assert(binIndex.z >= 0);
            assert(binIndex.x < SHAPE_CONTEXT_HORIZONTAL_SLICE_COUNT);
            assert(binIndex.y < SHAPE_CONTEXT_VERTICAL_SLICE_COUNT);
            assert(binIndex.z < SHAPE_CONTEXT_LAYER_COUNT);

            // 2. Compute sample weight
            float binVolume = computeSingleBinVolume(binIndex.y, binIndex.z, minSupportRadius.at(descriptorIndex), maxSupportRadius.at(descriptorIndex));

            // Volume can't be 0, and should be less than the volume of the support volume
            assert(binVolume > 0);
            assert(binVolume < (4.0f / 3.0f) * M_PI * maxSupportRadius.at(descriptorIndex) * maxSupportRadius.at(descriptorIndex) * maxSupportRadius.at(descriptorIndex));

            float sampleWeight = 1.0f / (pointDensityArray.content[sampleIndex] * std::cbrt(binVolume));

            // 3. Increment appropriate bin
            unsigned int index =
                    binIndex.x * SHAPE_CONTEXT_LAYER_COUNT * SHAPE_CONTEXT_VERTICAL_SLICE_COUNT +
                    binIndex.y * SHAPE_CONTEXT_LAYER_COUNT +
                    binIndex.z;
            assert(index < elementsPerShapeContextDescriptor);
            assert(!isnan(sampleWeight));
            descriptors.content[descriptorIndex].contents[index] += sampleWeight;
        }
    }
}

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

    ShapeDescriptor::cpu::array<uint32_t> pointCountArray = ShapeDescriptor::computePointDensities(pointDensityRadius, pointCloud);

    std::chrono::milliseconds pointCountingDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - pointCountingStart);

    // -- 3DSC Generation --
    auto generationStart = std::chrono::steady_clock::now();

    createDescriptors(
            imageOrigins,
            pointCloud,
            descriptors,
            pointCountArray,
            pointCloud.pointCount,
            minSupportRadius,
            maxSupportRadius);

    std::chrono::milliseconds generationDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - generationStart);

    // -- Cleanup --
    ShapeDescriptor::free(pointCountArray);

    std::chrono::milliseconds totalExecutionDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - totalExecutionTimeStart);

    if(executionTimes != nullptr) {
        executionTimes->totalExecutionTimeSeconds = double(totalExecutionDuration.count()) / 1000.0;
        executionTimes->generationTimeSeconds = double(generationDuration.count()) / 1000.0;
        executionTimes->pointCountingTimeSeconds = double(pointCountingDuration.count()) / 1000.0;
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


