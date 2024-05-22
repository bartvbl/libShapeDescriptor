#include <shapeDescriptor/shapeDescriptor.h>
#include <glm/glm.hpp>

std::vector<ShapeDescriptor::LocalReferenceFrame> ShapeDescriptor::internal::computeSHOTReferenceFrames(
        const ShapeDescriptor::cpu::PointCloud& pointCloud,
        const ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint>& imageOrigins,
        const std::vector<float>& maxSupportRadius) {
    std::vector<ShapeDescriptor::LocalReferenceFrame> referenceFrames(imageOrigins.length);
    std::vector<float> referenceWeightsZ(imageOrigins.length);
    std::vector<glm::mat3> covarianceMatrices(imageOrigins.length, glm::mat3(1.0));
    std::vector<int32_t> directionVotes(2 * imageOrigins.length);

    // Compute normalisation factors Z
    for(uint32_t pointIndex = 0; pointIndex < pointCloud.pointCount; pointIndex++) {
        ShapeDescriptor::cpu::float3 point = pointCloud.vertices[pointIndex];
        for(uint32_t originIndex = 0; originIndex < imageOrigins.length; originIndex++) {
            ShapeDescriptor::cpu::float3 origin = imageOrigins.content[originIndex].vertex;
            float distance = length(point - origin);
            if(distance <= maxSupportRadius.at(originIndex)) {
                referenceWeightsZ.at(originIndex) += maxSupportRadius.at(originIndex) - distance;
            }
        }
    }

    // Compute covariance matrices
    for(uint32_t pointIndex = 0; pointIndex < pointCloud.pointCount; pointIndex++) {
        ShapeDescriptor::cpu::float3 point = pointCloud.vertices[pointIndex];
        for(uint32_t originIndex = 0; originIndex < imageOrigins.length; originIndex++) {
            ShapeDescriptor::cpu::float3 origin = imageOrigins.content[originIndex].vertex;
            ShapeDescriptor::cpu::float3 pointDelta = point - origin;
            float distance = length(pointDelta);
            if(distance <= maxSupportRadius.at(originIndex)) {
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

    // Compute initial eigenvalues
    for(uint32_t originIndex = 0; originIndex < imageOrigins.length; originIndex++) {
        glm::mat3 &matrixToConvert = covarianceMatrices.at(originIndex);
        std::array<ShapeDescriptor::cpu::float3, 3> convertedMatrix = {
                ShapeDescriptor::cpu::float3{matrixToConvert[0][0], matrixToConvert[0][1], matrixToConvert[0][2]},
                ShapeDescriptor::cpu::float3{matrixToConvert[1][0], matrixToConvert[1][1], matrixToConvert[1][2]},
                ShapeDescriptor::cpu::float3{matrixToConvert[2][0], matrixToConvert[2][1], matrixToConvert[2][2]}
        };
        std::array<ShapeDescriptor::cpu::float3, 3> eigenVectors = ShapeDescriptor::internal::computeEigenVectors(convertedMatrix);
        referenceFrames.at(originIndex) = {eigenVectors.at(0), eigenVectors.at(1), eigenVectors.at(2)};
    }

    // Compute directional votes
    for(uint32_t pointIndex = 0; pointIndex < pointCloud.pointCount; pointIndex++) {
        ShapeDescriptor::cpu::float3 point = pointCloud.vertices[pointIndex];
        for(uint32_t originIndex = 0; originIndex < imageOrigins.length; originIndex++) {
            ShapeDescriptor::cpu::float3 origin = imageOrigins.content[originIndex].vertex;
            ShapeDescriptor::cpu::float3 pointDelta = point - origin;
            ShapeDescriptor::LocalReferenceFrame& frame = referenceFrames.at(originIndex);
            float dotX = dot(frame.xAxis, pointDelta);
            float dotZ = dot(frame.zAxis, pointDelta);
            directionVotes.at(2 * originIndex + 0) += dotX > 0 ? 1 : -1;
            directionVotes.at(2 * originIndex + 1) += dotZ > 0 ? 1 : -1;
        }
    }

    // Apply direction corrections
    for(uint32_t originIndex = 0; originIndex < imageOrigins.length; originIndex++) {
        ShapeDescriptor::LocalReferenceFrame& frame = referenceFrames.at(originIndex);
        if(directionVotes.at(2 * originIndex + 0) < 0) {
            frame.xAxis *= -1;
        }
        if(directionVotes.at(2 * originIndex + 1) < 0) {
            frame.zAxis *= -1;
        }
        frame.yAxis = cross(frame.xAxis, frame.zAxis);
    }

    return referenceFrames;
}