#include <shapeDescriptor/shapeDescriptor.h>
#include <shapeDescriptor/descriptors/ShapeContextGenerator.h>

#include <chrono>
#include <iostream>
#include <cmath>
#include <cstring>
#include <glm/glm.hpp>

std::vector<ShapeDescriptor::LocalReferenceFrame> computeUSCReferenceFrames(
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
                float relativeDistance = distance * referenceWeightsZ.at(originIndex);
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

ShapeDescriptor::cpu::array<ShapeDescriptor::UniqueShapeContextDescriptor> ShapeDescriptor::generalUniqueShapeContextMultiRadius(
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
    std::memset(descriptors.content, 0, descriptors.length * sizeof(UniqueShapeContextDescriptor::contents));

    // -- Point Count Computation --
    std::chrono::time_point pointCountingStart = std::chrono::steady_clock::now();

    std::vector<uint32_t> pointCountArray = ShapeDescriptor::computePointDensities(pointDensityRadius, pointCloud);

    std::chrono::milliseconds pointCountingDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - pointCountingStart);

    // -- USC Generation --
    auto generationStart = std::chrono::steady_clock::now();

    std::vector<ShapeDescriptor::LocalReferenceFrame> referenceFrames = computeUSCReferenceFrames(pointCloud, imageOrigins, maxSupportRadius);

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

ShapeDescriptor::cpu::array<ShapeDescriptor::UniqueShapeContextDescriptor> ShapeDescriptor::generalUniqueShapeContextDescriptors(
        ShapeDescriptor::cpu::PointCloud pointCloud,
        ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> imageOrigins,
        float pointDensityRadius,
        float minSupportRadius,
        float maxSupportRadius,
        ShapeDescriptor::SCExecutionTimes* executionTimes) {
    std::vector<float> minRadii(imageOrigins.length, minSupportRadius);
    std::vector<float> maxRadii(imageOrigins.length, maxSupportRadius);
    return generalUniqueShapeContextMultiRadius(pointCloud, imageOrigins, pointDensityRadius, minRadii, maxRadii, executionTimes);
}


