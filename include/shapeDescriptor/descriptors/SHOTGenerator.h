#pragma once

#include <shapeDescriptor/shapeDescriptor.h>

namespace ShapeDescriptor {

    template<typename SHOTDescriptor>
    cpu::array<SHOTDescriptor> generateSHOTDescriptorsMultiRadius(
            cpu::PointCloud cloud,
            cpu::array<OrientedPoint> descriptorOrigins,
            const std::vector<float>& supportRadii,
            SHOTExecutionTimes* executionTimes = nullptr) {
        ShapeDescriptor::cpu::array<SHOTDescriptor> descriptors{};


        std::vector<ShapeDescriptor::LocalReferenceFrame> referenceFrames(descriptorOrigins.length);
        std::vector<float> referenceWeightsZ(descriptorOrigins.length);
        std::vector<glm::mat3> covarianceMatrices(descriptorOrigins.length, glm::mat3(1.0));
        std::vector<int32_t> directionVotes(2 * descriptorOrigins.length);

        // Compute normalisation factors Z
        for(uint32_t pointIndex = 0; pointIndex < cloud.pointCount; pointIndex++) {
            ShapeDescriptor::cpu::float3 point = cloud.vertices[pointIndex];
            for(uint32_t originIndex = 0; originIndex < descriptorOrigins.length; originIndex++) {
                ShapeDescriptor::cpu::float3 origin = descriptorOrigins.content[originIndex].vertex;
                float distance = length(point - origin);
                if(distance <= supportRadii.at(originIndex)) {
                    referenceWeightsZ.at(originIndex) += supportRadii.at(originIndex) - distance;
                }
            }
        }

        // Compute covariance matrices
        for(uint32_t pointIndex = 0; pointIndex < cloud.pointCount; pointIndex++) {
            ShapeDescriptor::cpu::float3 point = cloud.vertices[pointIndex];
            for(uint32_t originIndex = 0; originIndex < descriptorOrigins.length; originIndex++) {
                ShapeDescriptor::cpu::float3 origin = descriptorOrigins.content[originIndex].vertex;
                ShapeDescriptor::cpu::float3 pointDelta = point - origin;
                float distance = length(pointDelta);
                if(distance <= supportRadii.at(originIndex)) {
                    glm::mat3 covarianceDeltaTransposed = {
                            pointDelta.x, pointDelta.y, pointDelta.z,
                            0, 0, 0,
                            0, 0, 0
                    };
                    glm::mat3 covarianceDelta = glm::transpose(covarianceDeltaTransposed);
                    float relativeDistance = distance * (1.0f / referenceWeightsZ.at(originIndex));
                    covarianceMatrices.at(originIndex) += relativeDistance * covarianceDelta * covarianceDeltaTransposed;
                }
            }
        }



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