#pragma once

#include <shapeDescriptor/shapeDescriptor.h>
#include "ShapeContextGenerator.h"

namespace ShapeDescriptor {

    namespace internal {
        template<uint32_t ELEVATION_DIVISIONS = 2, uint32_t RADIAL_DIVISIONS = 2, uint32_t AZIMUTH_DIVISIONS = 8, uint32_t INTERNAL_HISTOGRAM_BINS = 11>
        inline void incrementSHOTBin(ShapeDescriptor::SHOTDescriptor<ELEVATION_DIVISIONS, RADIAL_DIVISIONS, AZIMUTH_DIVISIONS, INTERNAL_HISTOGRAM_BINS>& descriptor, uint32_t elevationBinIndex, uint32_t radialBinIndex, uint32_t azimuthBinIndex, uint32_t histogramBinIndex, float contribution) {
            assert(elevationBinIndex < ELEVATION_DIVISIONS);
            assert(radialBinIndex < RADIAL_DIVISIONS);
            assert(azimuthBinIndex < AZIMUTH_DIVISIONS);
            assert(histogramBinIndex < INTERNAL_HISTOGRAM_BINS);
            uint32_t descriptorBinIndex =
                    elevationBinIndex * RADIAL_DIVISIONS * AZIMUTH_DIVISIONS * INTERNAL_HISTOGRAM_BINS
                  + radialBinIndex * AZIMUTH_DIVISIONS * INTERNAL_HISTOGRAM_BINS
                  + azimuthBinIndex * INTERNAL_HISTOGRAM_BINS
                  + histogramBinIndex;
            assert(descriptorBinIndex < ELEVATION_DIVISIONS * RADIAL_DIVISIONS * AZIMUTH_DIVISIONS * INTERNAL_HISTOGRAM_BINS);
            descriptor.contents[descriptorBinIndex] += contribution;
        }

        template<uint32_t ELEVATION_DIVISIONS = 2, uint32_t RADIAL_DIVISIONS = 2, uint32_t AZIMUTH_DIVISIONS = 8, uint32_t INTERNAL_HISTOGRAM_BINS = 11>
        void computeGeneralisedSHOTDescriptor(
                const ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> descriptorOrigins,
                const ShapeDescriptor::cpu::PointCloud pointCloud,
                ShapeDescriptor::cpu::array<ShapeDescriptor::SHOTDescriptor<ELEVATION_DIVISIONS, RADIAL_DIVISIONS, AZIMUTH_DIVISIONS, INTERNAL_HISTOGRAM_BINS>> descriptors,
                const std::vector<ShapeDescriptor::LocalReferenceFrame>& localReferenceFrames,
                const std::vector<float>& supportRadii)
        {
            const size_t elementsPerShapeContextDescriptor = RADIAL_DIVISIONS * ELEVATION_DIVISIONS * RADIAL_DIVISIONS;

            for(uint32_t descriptorIndex = 0; descriptorIndex < descriptors.length; descriptorIndex++) {
                const ShapeDescriptor::cpu::float3 vertex = descriptorOrigins.content[descriptorIndex].vertex;
                const ShapeDescriptor::cpu::float3 normal = descriptorOrigins.content[descriptorIndex].normal;

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
                    if (distanceToVertex > supportRadii.at(descriptorIndex)) {
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

                    const ShapeDescriptor::cpu::float3 sampleNormal = normalize(pointCloud.normals[sampleIndex]);
                    float normalCosine = dot(sampleNormal, normalize(normal));

                    // For the interpolations we'll use the order used in the paper
                    // a) Interpolation on normal cosines
                    float cosineHistogramPosition = clamp(((0.5f * normalCosine) + 0.5f) * float(INTERNAL_HISTOGRAM_BINS), 0.0f, float(INTERNAL_HISTOGRAM_BINS));
                    uint32_t cosineHistogramBinIndex = std::min(INTERNAL_HISTOGRAM_BINS - 1, uint32_t(cosineHistogramPosition));
                    float cosineHistogramDelta = cosineHistogramPosition - (float(cosineHistogramBinIndex) + 0.5f);
                    uint32_t cosineHistogramNeighbourBinIndex;
                    if(cosineHistogramDelta >= 0) {
                        cosineHistogramNeighbourBinIndex = (cosineHistogramBinIndex + 1) % INTERNAL_HISTOGRAM_BINS;
                    } else if(cosineHistogramDelta < 0 && cosineHistogramBinIndex > 0) {
                        cosineHistogramNeighbourBinIndex = cosineHistogramBinIndex - 1;
                    } else {
                        cosineHistogramNeighbourBinIndex = INTERNAL_HISTOGRAM_BINS - 1;
                    }
                    float cosineHistogramBinContribution = std::abs(cosineHistogramDelta);
                    float cosineHistogramNeighbourBinContribution = 1.0f - cosineHistogramBinContribution;


                    // b) Interpolation on azimuth
                    float azimuthAnglePosition = (internal::absoluteAngle(horizontalDirection.y, horizontalDirection.x) / (2.0f * float(M_PI))) * float(AZIMUTH_DIVISIONS);
                    if(azimuthAnglePosition < 0) {
                        azimuthAnglePosition += float(AZIMUTH_DIVISIONS);
                    } else if(azimuthAnglePosition >= float(AZIMUTH_DIVISIONS)) {
                        azimuthAnglePosition -= float(AZIMUTH_DIVISIONS);
                    }
                    uint32_t azimuthBinIndex = std::min(AZIMUTH_DIVISIONS - 1, uint32_t(azimuthAnglePosition));
                    float azimuthHistogramDelta = azimuthAnglePosition - (float(azimuthBinIndex) + 0.5f);
                    uint32_t azimuthNeighbourBinIndex;
                    if(azimuthHistogramDelta >= 0) {
                        azimuthNeighbourBinIndex = (azimuthBinIndex + 1) % AZIMUTH_DIVISIONS;
                    } else if(azimuthHistogramDelta < 0 && azimuthBinIndex > 0) {
                        azimuthNeighbourBinIndex = azimuthBinIndex - 1;
                    } else {
                        azimuthNeighbourBinIndex = AZIMUTH_DIVISIONS - 1;
                    }
                    float azimuthBinContribution = std::abs(azimuthHistogramDelta);
                    float azimuthNeighbourBinContribution = 1.0f - azimuthBinContribution;


                    // c) Interpolation on elevation


                    float verticalAngle = std::fmod(
                            internal::absoluteAngle(verticalDirection.y, verticalDirection.x) + (float(M_PI) / 2.0f),
                            2.0f * float(M_PI));
                    uint32_t verticalIndex = unsigned((verticalAngle / M_PI) * float(ELEVATION_DIVISIONS)) % ELEVATION_DIVISIONS;

                    float sampleDistance = length(relativeSamplePoint);
                    const float distancePerSlice = supportRadii.at(descriptorIndex) / float(RADIAL_DIVISIONS);
                    uint32_t layerIndex = std::min<uint32_t>(RADIAL_DIVISIONS, uint32_t(sampleDistance / distancePerSlice));

                    uint3 binIndex = {horizontalIndex, verticalIndex, layerIndex};
                    assert(binIndex.x < AZIMUTH_DIVISIONS);
                    assert(binIndex.y < ELEVATION_DIVISIONS);
                    assert(binIndex.z < RADIAL_DIVISIONS);




                    // 3. Increment appropriate bin
                    unsigned int index =
                            binIndex.x * RADIAL_DIVISIONS * ELEVATION_DIVISIONS +
                            binIndex.y * RADIAL_DIVISIONS +
                            binIndex.z;
                    assert(index < elementsPerShapeContextDescriptor);
                    //descriptors.content[descriptorIndex].contents[index] += sampleWeight;
                }
            }
        }
    }

    template<uint32_t ELEVATION_DIVISIONS = 2, uint32_t RADIAL_DIVISIONS = 2, uint32_t AZIMUTH_DIVISIONS = 8, uint32_t INTERNAL_HISTOGRAM_BINS = 11>
    cpu::array<ShapeDescriptor::SHOTDescriptor<ELEVATION_DIVISIONS, RADIAL_DIVISIONS, AZIMUTH_DIVISIONS, INTERNAL_HISTOGRAM_BINS>> generateSHOTDescriptorsMultiRadius(
            const cpu::PointCloud cloud,
            const cpu::array<OrientedPoint> descriptorOrigins,
            const std::vector<float>& supportRadii,
            SHOTExecutionTimes* executionTimes = nullptr) {
        ShapeDescriptor::cpu::array<ShapeDescriptor::SHOTDescriptor<ELEVATION_DIVISIONS, RADIAL_DIVISIONS, AZIMUTH_DIVISIONS, INTERNAL_HISTOGRAM_BINS>> descriptors(descriptorOrigins.length);

        std::vector<ShapeDescriptor::LocalReferenceFrame> referenceFrames = ShapeDescriptor::internal::computeSHOTReferenceFrames(cloud, descriptorOrigins, supportRadii);

        internal::computeGeneralisedSHOTDescriptor<ELEVATION_DIVISIONS, RADIAL_DIVISIONS, AZIMUTH_DIVISIONS, INTERNAL_HISTOGRAM_BINS>(descriptorOrigins, cloud, descriptors, referenceFrames, supportRadii);

        return descriptors;
    }

    template<uint32_t ELEVATION_DIVISIONS = 2, uint32_t RADIAL_DIVISIONS = 2, uint32_t AZIMUTH_DIVISIONS = 8, uint32_t INTERNAL_HISTOGRAM_BINS = 11>
    cpu::array<ShapeDescriptor::SHOTDescriptor<ELEVATION_DIVISIONS, RADIAL_DIVISIONS, AZIMUTH_DIVISIONS, INTERNAL_HISTOGRAM_BINS>> generateSHOTDescriptors(
            cpu::PointCloud cloud,
            cpu::array<OrientedPoint> descriptorOrigins,
            float supportRadius,
            SHOTExecutionTimes* executionTimes = nullptr) {
        std::vector<float> radii(descriptorOrigins.length, supportRadius);

        return generateSHOTDescriptorsMultiRadius<ShapeDescriptor::SHOTDescriptor<ELEVATION_DIVISIONS, RADIAL_DIVISIONS, AZIMUTH_DIVISIONS, INTERNAL_HISTOGRAM_BINS>>(cloud, descriptorOrigins, radii, executionTimes);
    }
}