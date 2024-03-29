#pragma once
#include <shapeDescriptor/shapeDescriptor.h>

namespace ShapeDescriptor {
    namespace internal {
        __SD_HOST_DEVICE inline float
        computeLayerDistance(float minSupportRadius, float maxSupportRadius, short layerIndex) {
            // Avoiding zero divisions
            if (minSupportRadius == 0) {
                minSupportRadius = 0.000001f;
            }
            return std::exp(
                    (std::log(minSupportRadius))
                    + ((float(layerIndex) / float(SHAPE_CONTEXT_LAYER_COUNT))
                       * std::log(float(maxSupportRadius) / float(minSupportRadius))));
        }


        __SD_HOST_DEVICE inline float computeWedgeSegmentVolume(short verticalBinIndex, float radius) {
            const float verticalAngleStep = 1.0f / float(SHAPE_CONTEXT_VERTICAL_SLICE_COUNT);
            float binStartAngle = float(verticalBinIndex) * verticalAngleStep;
            float binEndAngle = float(verticalBinIndex + 1) * verticalAngleStep;

            float scaleFraction = (2.0f * float(M_PI) * radius * radius * radius)
                                  / (3.0f * float(SHAPE_CONTEXT_HORIZONTAL_SLICE_COUNT));
            return scaleFraction * (std::cos(binStartAngle) - std::cos(binEndAngle));
        }

        __SD_HOST_DEVICE inline float
        computeBinVolume(short verticalBinIndex, short layerIndex, float minSupportRadius, float maxSupportRadius) {
            // The wedge segment computation goes all the way from the center to the edge of the sphere
            // Since we also have a minimum support radius, we need to cut out the volume of the centre part
            float binEndRadius = computeLayerDistance(minSupportRadius, maxSupportRadius, layerIndex + 1);
            float binStartRadius = computeLayerDistance(minSupportRadius, maxSupportRadius, layerIndex);

            float largeSupportRadiusVolume = computeWedgeSegmentVolume(verticalBinIndex, binEndRadius);
            float smallSupportRadiusVolume = computeWedgeSegmentVolume(verticalBinIndex, binStartRadius);

            return largeSupportRadiusVolume - smallSupportRadiusVolume;
        }

        __SD_HOST_DEVICE inline float absoluteAngle(float y, float x) {
            float absoluteAngle = std::atan2(y, x);
            return absoluteAngle < 0 ? absoluteAngle + (2.0f * float(M_PI)) : absoluteAngle;
        }
    }

    template<uint32_t horizontalSliceCount, uint32_t verticalSliceCount, uint32_t sliceCount>
    void computeGeneralShapeContextDescriptors(
            ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> descriptorOrigins,
            ShapeDescriptor::cpu::PointCloud pointCloud,
            ShapeDescriptor::cpu::array<ShapeDescriptor::GeneralShapeContextDescriptor<horizontalSliceCount, verticalSliceCount, sliceCount>> descriptors,
            std::vector<unsigned int> pointDensityArray,
            const std::vector<ShapeDescriptor::LocalReferenceFrame>& localReferenceFrames,
            const std::vector<float>& minSupportRadius,
            const std::vector<float>& maxSupportRadius)
    {
        const size_t elementsPerShapeContextDescriptor = horizontalSliceCount * verticalSliceCount * sliceCount;

        for(uint32_t descriptorIndex = 0; descriptorIndex < descriptors.length; descriptorIndex++) {
            std::array<float, sliceCount> layerDistances;
            for(int i = 0; i < sliceCount; i++) {
                layerDistances.at(i) = internal::computeLayerDistance(minSupportRadius.at(descriptorIndex), maxSupportRadius.at(descriptorIndex), i + 1);
            }

            const ShapeDescriptor::cpu::float3 vertex = descriptorOrigins[descriptorIndex].vertex;

            for(float & content : descriptors.content[descriptorIndex].contents) {
                content = 0;
            }

            for (unsigned int sampleIndex = 0; sampleIndex < pointCloud.pointCount; sampleIndex++) {
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
                const ShapeDescriptor::cpu::float3 relativeSamplePoint = {
                        dot(localReferenceFrames.at(descriptorIndex).xAxis, translated),
                        dot(localReferenceFrames.at(descriptorIndex).yAxis, translated),
                        dot(localReferenceFrames.at(descriptorIndex).zAxis, translated)
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

                float horizontalAngle = internal::absoluteAngle(horizontalDirection.y, horizontalDirection.x);
                short horizontalIndex = unsigned((horizontalAngle / (2.0f * float(M_PI))) * float(horizontalSliceCount)) % horizontalSliceCount;

                float verticalAngle = std::fmod(
                        internal::absoluteAngle(verticalDirection.y, verticalDirection.x) + (float(M_PI) / 2.0f),
                        2.0f * float(M_PI));
                short verticalIndex = unsigned((verticalAngle / M_PI) * float(verticalSliceCount)) % verticalSliceCount;

                float sampleDistance = length(relativeSamplePoint);
                short layerIndex = 0;

                for (; layerIndex < sliceCount; layerIndex++) {
                    float nextSliceEnd = layerDistances.at(layerIndex);
                    if (sampleDistance <= nextSliceEnd) {
                        break;
                    }
                }

                // Rounding errors can cause it to exceed its allowed bounds in specific cases
                // Of course, on the off chance something is wrong after all,
                // the assertions further down should trip. So we only handle the single
                // edge case where layerIndex went one over.
                if (layerIndex == sliceCount) {
                    layerIndex--;
                }

                short3 binIndex = {horizontalIndex, verticalIndex, layerIndex};
                assert(binIndex.x >= 0);
                assert(binIndex.y >= 0);
                assert(binIndex.z >= 0);
                assert(binIndex.x < horizontalSliceCount);
                assert(binIndex.y < verticalSliceCount);
                assert(binIndex.z < sliceCount);

                // 2. Compute sample weight
                float binVolume = internal::computeBinVolume(binIndex.y, binIndex.z, minSupportRadius.at(descriptorIndex), maxSupportRadius.at(descriptorIndex));

                // Volume can't be 0, and should be less than the volume of the support volume
                assert(binVolume > 0);
                assert(binVolume < (4.0f / 3.0f) * M_PI * maxSupportRadius.at(descriptorIndex) * maxSupportRadius.at(descriptorIndex) * maxSupportRadius.at(descriptorIndex));

                float sampleWeight = 1.0f / (pointDensityArray.at(sampleIndex) * std::cbrt(binVolume));

                // 3. Increment appropriate bin
                unsigned int index =
                        binIndex.x * sliceCount * verticalSliceCount +
                        binIndex.y * sliceCount +
                        binIndex.z;
                assert(index < elementsPerShapeContextDescriptor);
                assert(!isnan(sampleWeight));
                assert(!isinf(sampleWeight));
                descriptors.content[descriptorIndex].contents[index] += sampleWeight;
            }
        }
    }
}

