#pragma once

#include <shapeDescriptor/shapeDescriptor.h>

namespace ShapeDescriptor {

    namespace internal {
        template<uint32_t ELEVATION_DIVISIONS = 2, uint32_t RADIAL_DIVISIONS = 2, uint32_t AZIMUTH_DIVISIONS = 8, uint32_t INTERNAL_HISTOGRAM_BINS = 11>
        void computeGeneralisedSHOTDescriptor(
                ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> descriptorOrigins,
                ShapeDescriptor::cpu::PointCloud pointCloud,
                ShapeDescriptor::cpu::array<ShapeDescriptor::SHOTDescriptor<ELEVATION_DIVISIONS, RADIAL_DIVISIONS, AZIMUTH_DIVISIONS, INTERNAL_HISTOGRAM_BINS>> descriptors,
                std::vector<unsigned int> pointDensityArray,
                const std::vector<ShapeDescriptor::LocalReferenceFrame>& localReferenceFrames,
                const std::vector<float>& supportRadii)
        {
            const size_t elementsPerShapeContextDescriptor = RADIAL_DIVISIONS * ELEVATION_DIVISIONS * RADIAL_DIVISIONS;

            for(uint32_t descriptorIndex = 0; descriptorIndex < descriptors.length; descriptorIndex++) {
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

                    float horizontalAngle = internal::absoluteAngle(horizontalDirection.y, horizontalDirection.x);
                    uint32_t horizontalIndex = unsigned((horizontalAngle / (2.0f * float(M_PI))) * float(AZIMUTH_DIVISIONS)) % AZIMUTH_DIVISIONS;

                    float verticalAngle = std::fmod(
                            internal::absoluteAngle(verticalDirection.y, verticalDirection.x) + (float(M_PI) / 2.0f),
                            2.0f * float(M_PI));
                    uint32_t verticalIndex = unsigned((verticalAngle / M_PI) * float(ELEVATION_DIVISIONS)) % ELEVATION_DIVISIONS;

                    float sampleDistance = length(relativeSamplePoint);
                    const float distancePerSlice = supportRadii.at(descriptorIndex) / float(RADIAL_DIVISIONS);
                    uint32_t layerIndex = std::min<uint32_t>(RADIAL_DIVISIONS, uint32_t(sampleDistance / distancePerSlice));

                    uint3 binIndex = {horizontalIndex, verticalIndex, layerIndex};
                    assert(binIndex.x >= 0);
                    assert(binIndex.y >= 0);
                    assert(binIndex.z >= 0);
                    assert(binIndex.x < AZIMUTH_DIVISIONS);
                    assert(binIndex.y < ELEVATION_DIVISIONS);
                    assert(binIndex.z < RADIAL_DIVISIONS);


                    // 3. Increment appropriate bin
                    unsigned int index =
                            binIndex.x * RADIAL_DIVISIONS * ELEVATION_DIVISIONS +
                            binIndex.y * RADIAL_DIVISIONS +
                            binIndex.z;
                    assert(index < elementsPerShapeContextDescriptor);
                    descriptors.content[descriptorIndex].contents[index] += sampleWeight;
                }
            }
        }
    }

    template<uint32_t ELEVATION_DIVISIONS = 2, uint32_t RADIAL_DIVISIONS = 2, uint32_t AZIMUTH_DIVISIONS = 8, uint32_t INTERNAL_HISTOGRAM_BINS = 11>
    cpu::array<ShapeDescriptor::SHOTDescriptor<ELEVATION_DIVISIONS, RADIAL_DIVISIONS, AZIMUTH_DIVISIONS, INTERNAL_HISTOGRAM_BINS>> generateSHOTDescriptorsMultiRadius(
            cpu::PointCloud cloud,
            cpu::array<OrientedPoint> descriptorOrigins,
            const std::vector<float>& supportRadii,
            SHOTExecutionTimes* executionTimes = nullptr) {
        ShapeDescriptor::cpu::array<ShapeDescriptor::SHOTDescriptor<ELEVATION_DIVISIONS, RADIAL_DIVISIONS, AZIMUTH_DIVISIONS, INTERNAL_HISTOGRAM_BINS>> descriptors(descriptorOrigins.length);

        std::vector<ShapeDescriptor::LocalReferenceFrame> referenceFrames = ShapeDescriptor::internal::computeSHOTReferenceFrames(cloud, descriptorOrigins, supportRadii);

        internal::computeGeneralisedSHOTDescriptor<ELEVATION_DIVISIONS, RADIAL_DIVISIONS, AZIMUTH_DIVISIONS, INTERNAL_HISTOGRAM_BINS>(descriptorOrigins, cloud, descriptors, referenceFrames, supportRadii);

        return descriptors;
    }

    template<typename SHOTDescriptor>
    cpu::array<SHOTDescriptor> generateSHOTDescriptors(
            cpu::PointCloud cloud,
            cpu::array<OrientedPoint> descriptorOrigins,
            float supportRadius,
            SHOTExecutionTimes* executionTimes = nullptr) {
        std::vector<float> radii(descriptorOrigins.length, supportRadius);

        return generateSHOTDescriptorsMultiRadius<SHOTDescriptor>(cloud, descriptorOrigins, radii, executionTimes);
    }
}