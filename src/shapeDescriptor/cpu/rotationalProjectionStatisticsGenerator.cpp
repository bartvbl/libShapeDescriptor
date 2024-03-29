#include <shapeDescriptor/shapeDescriptor.h>
#include <glm/glm.hpp>
#include <glm/ext/matrix_transform.hpp>

// 1. Compute local reference frame
//      - Compute a list of all points in the support radius
//      - For each point in the support radius (shaped as a sphere):
//          -
// 2. For several rotations around the x, y, and z-axis:
//      3. Project on to different planes
//      4. Compute 5x5 histogram fir each
//      5. Normlise matrices
//      6. Compute vector of statistics

// From: Realtime collission detection, Christer Ericson, Morgan Kaufman 2005
ShapeDescriptor::cpu::float3 computeClosestPointToTriangle(ShapeDescriptor::cpu::float3 p,
                                                           ShapeDescriptor::cpu::float3 a,
                                                           ShapeDescriptor::cpu::float3 b,
                                                           ShapeDescriptor::cpu::float3 c) {
    ShapeDescriptor::cpu::float3 ab = b - a;
    ShapeDescriptor::cpu::float3 ac = c - a;
    ShapeDescriptor::cpu::float3 ap = p - a;
    float d1 = dot(ab, ap);
    float d2 = dot(ac, ap);

    if(d1 <= 0 && d2 <= 0) {
        return a;
    }

    ShapeDescriptor::cpu::float3 bp = p - b;
    float d3 = dot(ab, bp);
    float d4 = dot(ac, bp);
    if(d3 >= 0 && d4 <= d3) {
        return b;
    }

    float vc = d1 * d4 - d3 * d2;
    if(vc <= 0 && d1 >= 0 && d3 <= 0) {
        float v = d1 / (d1 - d3);
        return a + v * ab;
    }

    ShapeDescriptor::cpu::float3 cp = p - c;
    float d5 = dot(ab, cp);
    float d6 = dot(ac, cp);
    if(d6 >= 0 && d5 <= d6) {
        return c;
    }

    float vb = d5 * d2 - d1 * d6;
    if(vb <= 0 && d2 >= 0 && d6 <= 0) {
        float w = d2 / (d2 - d6);
        return a + w * ac;
    }

    float va = d3 * d6 - d5 * d4;
    if(va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) {
        float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + w * (c - b);
    }

    float denom = 1.0f / (va + vb + vc);
    float v = vb * denom;
    float w = vc * denom;
    return a + ab * v + ac * w;
}

bool sphereTriangleIntersectionTest(ShapeDescriptor::cpu::float3 sphereCentre,
                                    float sphereRadius,
                                    ShapeDescriptor::cpu::float3 triangleVertex0,
                                    ShapeDescriptor::cpu::float3 triangleVertex1,
                                    ShapeDescriptor::cpu::float3 triangleVertex2) {
    ShapeDescriptor::cpu::float3 closestPointOnTriangle = computeClosestPointToTriangle(sphereCentre, triangleVertex0, triangleVertex1, triangleVertex2);
    ShapeDescriptor::cpu::float3 sphereToClosestPoint = closestPointOnTriangle - sphereCentre;
    return dot(sphereToClosestPoint, sphereToClosestPoint) <= sphereRadius * sphereRadius;
}

inline glm::mat3 constructPartialMatrix(ShapeDescriptor::cpu::float3 delta1, ShapeDescriptor::cpu::float3 delta2) {
    glm::mat3 matrix(
        delta1.x * delta2.x,    delta1.x * delta2.y,    delta1.x * delta2.z,
        delta1.y * delta2.x,    delta1.y * delta2.y,    delta1.y * delta2.z,
        delta1.z * delta2.x,    delta1.z * delta2.y,    delta1.z * delta2.z
    );
    return matrix;
}

std::vector<std::array<ShapeDescriptor::cpu::float3, 3>> computeLocalReferenceFrames(ShapeDescriptor::cpu::Mesh mesh,
                                              ShapeDescriptor::cpu::array <ShapeDescriptor::OrientedPoint> referencePoints,
                                              float supportRadius,
                                              double& totalMeshArea) {
    std::vector<glm::mat3> matrices(referencePoints.length, glm::mat3(0));
    std::vector<double> supportTriangleAreas(referencePoints.length);
    std::vector<std::array<ShapeDescriptor::cpu::float3, 3>> eigenVectors(referencePoints.length);
    std::vector<std::array<double, 3>> accumulatedHValues(referencePoints.length);
    std::vector<std::array<ShapeDescriptor::cpu::float3, 3>> localReferenceFrames(referencePoints.length);

    totalMeshArea = 0;

    for(uint32_t index = 0; index < mesh.vertexCount; index += 3) {
        ShapeDescriptor::cpu::float3 vertex0 = mesh.vertices[index];
        ShapeDescriptor::cpu::float3 vertex1 = mesh.vertices[index + 1];
        ShapeDescriptor::cpu::float3 vertex2 = mesh.vertices[index + 2];

        float triangleArea = ShapeDescriptor::computeTriangleArea(vertex0, vertex1, vertex2);
        totalMeshArea += triangleArea;

        for(uint32_t referencePointIndex = 0; referencePointIndex < referencePoints.length; referencePointIndex++) {
            ShapeDescriptor::cpu::float3 referencePoint = referencePoints[referencePointIndex].vertex;
            if(sphereTriangleIntersectionTest(referencePoint, supportRadius, vertex0, vertex1, vertex2)) {

                glm::mat3 scatterMatrix(0.0);
                std::array<ShapeDescriptor::cpu::float3, 3> triangleCoordinates = {
                        vertex0, vertex1, vertex2
                };

                for(int j = 0; j < 3; j++) {
                    ShapeDescriptor::cpu::float3 p_ijDelta = triangleCoordinates[j] - referencePoint;
                    for(int k = 0; k < 3; k++) {
                        ShapeDescriptor::cpu::float3 p_ikDelta = triangleCoordinates[k] - referencePoint;
                        scatterMatrix += constructPartialMatrix(p_ijDelta, p_ikDelta);
                    }
                    scatterMatrix += constructPartialMatrix(p_ijDelta, p_ijDelta);
                }

                // The division is done later
                float weight1 = triangleArea;
                supportTriangleAreas.at(referencePointIndex) += triangleArea;
                float partialWeight2 = supportRadius - length(referencePoint - (vertex0 + vertex1 + vertex2) / 3.0f);
                float weight2 = partialWeight2 * partialWeight2;

                std::array<double, 3> hValues {0, 0, 0};
                float scaleFactor = weight1 * weight2 * (1.0f / 6.0f);
                for(uint32_t j = 0; j < 3; j++) {
                    ShapeDescriptor::cpu::float3 referencePointDelta = triangleCoordinates[j] - referencePoint;
                    hValues.at(0) += scaleFactor * referencePointDelta.x;
                    hValues.at(1) += scaleFactor * referencePointDelta.y;
                    hValues.at(2) += scaleFactor * referencePointDelta.z;
                }

                accumulatedHValues.at(referencePointIndex).at(0) += hValues.at(0);
                accumulatedHValues.at(referencePointIndex).at(1) += hValues.at(1);
                accumulatedHValues.at(referencePointIndex).at(2) += hValues.at(2);

                matrices.at(referencePointIndex) += (weight1 * weight2 * (1.0f / 12.0f) * scatterMatrix);
            }
        }
    }

    for(uint32_t referencePointIndex = 0; referencePointIndex < referencePoints.length; referencePointIndex++) {
        // We need to scale the matrices to the total area in the support radius to adhere to the specification
        double weight1Factor = (1.0 / supportTriangleAreas.at(referencePointIndex));
        matrices.at(referencePointIndex) *= weight1Factor;
        glm::mat3& matrixToConvert = matrices.at(referencePointIndex);
        std::array<ShapeDescriptor::cpu::float3, 3> convertedMatrix = {
                ShapeDescriptor::cpu::float3{matrixToConvert[0][0], matrixToConvert[0][1], matrixToConvert[0][2]},
                ShapeDescriptor::cpu::float3{matrixToConvert[1][0], matrixToConvert[1][1], matrixToConvert[1][2]},
                ShapeDescriptor::cpu::float3{matrixToConvert[2][0], matrixToConvert[2][1], matrixToConvert[2][2]}
        };
        eigenVectors.at(referencePointIndex) = ShapeDescriptor::internal::computeEigenVectors(convertedMatrix);

        // Implicitly doing a dot product
        std::array<double, 3> hCoefficients = accumulatedHValues.at(referencePointIndex);
        hCoefficients.at(0) *= weight1Factor;
        hCoefficients.at(1) *= weight1Factor;
        hCoefficients.at(2) *= weight1Factor;

        ShapeDescriptor::cpu::float3 v1 = eigenVectors.at(referencePointIndex).at(0);
        double h1 = hCoefficients.at(0) * v1.x + hCoefficients.at(1) * v1.y + hCoefficients.at(2) * v1.z;
        ShapeDescriptor::cpu::float3 v3 = eigenVectors.at(referencePointIndex).at(2);
        double h3 = hCoefficients.at(0) * v3.x + hCoefficients.at(1) * v3.y + hCoefficients.at(2) * v3.z;

        ShapeDescriptor::cpu::float3 v1_unambiguous = v1 * (std::signbit(h1) ? -1.0f : 1.0f);
        ShapeDescriptor::cpu::float3 v3_unambiguous = v3 * (std::signbit(h3) ? -1.0f : 1.0f);
        ShapeDescriptor::cpu::float3 v2_unambiguous = cross(v3, v1);

        localReferenceFrames.at(referencePointIndex) = { v1_unambiguous, v2_unambiguous, v3_unambiguous };
    }

    return localReferenceFrames;
}

struct PlaneBounds {
    float minX = std::numeric_limits<float>::max();
    float maxX = -std::numeric_limits<float>::max();
    float minY = std::numeric_limits<float>::max();
    float maxY = -std::numeric_limits<float>::max();
    float minZ = std::numeric_limits<float>::max();
    float maxZ = -std::numeric_limits<float>::max();
};

ShapeDescriptor::RoPSDescriptor computeRoPSDescriptor(const ShapeDescriptor::cpu::PointCloud cloud,
                                                      ShapeDescriptor::OrientedPoint &referencePoint,
                                                      std::array<ShapeDescriptor::cpu::float3, 3>& localReferenceFrame,
                                                      float supportRadius) {

    glm::mat3 lrfTransform(
              localReferenceFrame.at(0).x,  localReferenceFrame.at(1).x,    localReferenceFrame.at(2).x,
              localReferenceFrame.at(0).y,  localReferenceFrame.at(1).y,    localReferenceFrame.at(2).y,
              localReferenceFrame.at(0).z,  localReferenceFrame.at(1).z,    localReferenceFrame.at(2).z
    );

    ShapeDescriptor::RoPSDescriptor descriptor{};
    for(int i = 0; i < sizeof(descriptor) / 4; i++) {
        descriptor.contents[i] = 0;
    }

    constexpr int rotatedPointCloudCount = 3 * ROPS_NUM_ROTATIONS;
    constexpr int histogramCount = 3 * rotatedPointCloudCount;
    std::array<std::array<std::array<float, ROPS_HISTOGRAM_BINS>, ROPS_HISTOGRAM_BINS>, histogramCount> intermediateHistograms{};
    for(auto& intermediateHistogram : intermediateHistograms) {
        for(auto& histogramRow : intermediateHistogram) {
            for(float& histogramBin : histogramRow) {
                histogramBin = 0;
            }
        }
    }

    const float angleStep = (2.0f * M_PI) / float(ROPS_NUM_ROTATIONS);
    std::array<glm::mat3, rotatedPointCloudCount> rotationMatrices{};
    for(int i = 0; i < ROPS_NUM_ROTATIONS; i++) {
        float rotationAngle = float(i) * angleStep;
        // Order is important here and follows the one used in the paper
        rotationMatrices.at(ROPS_NUM_ROTATIONS * 0 + i) = glm::rotate(glm::mat4(1.0), rotationAngle, glm::vec3(1.0, 0.0, 0.0));
        rotationMatrices.at(ROPS_NUM_ROTATIONS * 1 + i) = glm::rotate(glm::mat4(1.0), rotationAngle, glm::vec3(0.0, 1.0, 0.0));
        rotationMatrices.at(ROPS_NUM_ROTATIONS * 2 + i) = glm::rotate(glm::mat4(1.0), rotationAngle, glm::vec3(0.0, 0.0, 1.0));
    }
    std::array<PlaneBounds, rotatedPointCloudCount> planeBounds;
    for(int i = 0; i < rotatedPointCloudCount; i++) {
        planeBounds.at(i) = PlaneBounds();
    }

    // Calculating bounds
    uint32_t pointCountInSupportRadius = 0;
    for(uint32_t vertexIndex = 0; vertexIndex < cloud.pointCount; vertexIndex++) {
        ShapeDescriptor::cpu::float3 point = cloud.vertices[vertexIndex];
        bool isInSupportRadius = length(point - referencePoint.vertex) <= supportRadius;
        if(isInSupportRadius) {
            pointCountInSupportRadius++;
            glm::vec3 lrfPoint = lrfTransform * glm::vec3(point.x, point.y, point.z);
            for(uint32_t matrixIndex = 0; matrixIndex < rotationMatrices.size(); matrixIndex++) {
                glm::vec3 transformedPoint = rotationMatrices.at(matrixIndex) * lrfPoint;
                PlaneBounds& bounds = planeBounds.at(matrixIndex);
                bounds.maxX = std::max(bounds.maxX, transformedPoint.x);
                bounds.maxY = std::max(bounds.maxY, transformedPoint.y);
                bounds.maxZ = std::max(bounds.maxZ, transformedPoint.z);
                bounds.minX = std::min(bounds.minX, transformedPoint.x);
                bounds.minY = std::min(bounds.minY, transformedPoint.y);
                bounds.minZ = std::min(bounds.minZ, transformedPoint.z);
            }
        }
    }

    // No points in support radius, so we return an empty descriptor
    if(pointCountInSupportRadius == 0) {
        return descriptor;
    }

    // Computing histograms
    for(uint32_t vertexIndex = 0; vertexIndex < cloud.pointCount; vertexIndex++) {
        ShapeDescriptor::cpu::float3 point = cloud.vertices[vertexIndex];
        bool isInSupportRadius = length(point - referencePoint.vertex) <= supportRadius;
        if(isInSupportRadius) {
            glm::vec3 lrfPoint = lrfTransform * glm::vec3(point.x, point.y, point.z);
            for(uint32_t matrixIndex = 0; matrixIndex < rotationMatrices.size(); matrixIndex++) {
                PlaneBounds& bounds = planeBounds.at(matrixIndex);
                glm::vec3 rotatedPoint = rotationMatrices.at(matrixIndex) * lrfPoint;

                int xIndex = int(float(ROPS_HISTOGRAM_BINS) * (rotatedPoint.x - bounds.minX) / (bounds.maxX - bounds.minX));
                int yIndex = int(float(ROPS_HISTOGRAM_BINS) * (rotatedPoint.y - bounds.minY) / (bounds.maxY - bounds.minY));
                int zIndex = int(float(ROPS_HISTOGRAM_BINS) * (rotatedPoint.z - bounds.minZ) / (bounds.maxZ - bounds.minZ));

                // Ensure that rounding errors do not cause out of range indices
                xIndex = clamp(xIndex, 0, ROPS_HISTOGRAM_BINS - 1);
                yIndex = clamp(yIndex, 0, ROPS_HISTOGRAM_BINS - 1);
                zIndex = clamp(zIndex, 0, ROPS_HISTOGRAM_BINS - 1);

                // xy plane
                intermediateHistograms.at(3 * matrixIndex + 0).at(xIndex).at(yIndex)++;

                // xz plane
                intermediateHistograms.at(3 * matrixIndex + 1).at(xIndex).at(zIndex)++;

                // yz plane
                intermediateHistograms.at(3 * matrixIndex + 2).at(yIndex).at(zIndex)++;
            }
        }
    }



    for(uint32_t histogramIndex = 0; histogramIndex < intermediateHistograms.size(); histogramIndex++) {
        std::array<std::array<float, ROPS_HISTOGRAM_BINS>, ROPS_HISTOGRAM_BINS>& histogram = intermediateHistograms.at(histogramIndex);

        // Normalise histogram
        float histogramSum = 0;
        for(uint32_t row = 0; row < ROPS_HISTOGRAM_BINS; row++) {
            for(uint32_t col = 0; col < ROPS_HISTOGRAM_BINS; col++) {
                histogramSum += histogram.at(col).at(row);
            }
        }
        for(uint32_t row = 0; row < ROPS_HISTOGRAM_BINS; row++) {
            for(uint32_t col = 0; col < ROPS_HISTOGRAM_BINS; col++) {
                histogram.at(col).at(row) /= histogramSum;
            }
        }

        // Compute iStrike and jStrike
        float iStrike = 0;
        float jStrike = 0;
        for(uint32_t row = 0; row < ROPS_HISTOGRAM_BINS; row++) {
            for(uint32_t col = 0; col < ROPS_HISTOGRAM_BINS; col++) {
                iStrike += float(col) * histogram.at(col).at(row);
                jStrike += float(row) * histogram.at(col).at(row);
            }
        }

        // Compute central moments and Shannon entropy
        float centralMoment_11 = 0;
        float centralMoment_21 = 0;
        float centralMoment_12 = 0;
        float centralMoment_22 = 0;
        float shannonEntropy = 0;
        for(uint32_t row = 0; row < ROPS_HISTOGRAM_BINS; row++) {
            for(uint32_t col = 0; col < ROPS_HISTOGRAM_BINS; col++) {
                float binValue = histogram.at(col).at(row);
                float iStrikeDelta = binValue - iStrike;
                float jStrikeDelta = binValue - jStrike;

                centralMoment_11 += iStrikeDelta * jStrikeDelta * binValue;
                centralMoment_21 += iStrikeDelta * iStrikeDelta * jStrikeDelta * binValue;
                centralMoment_12 += iStrikeDelta * jStrikeDelta * jStrikeDelta * binValue;
                centralMoment_22 += iStrikeDelta * iStrikeDelta * jStrikeDelta * jStrikeDelta * binValue;
                if(binValue > 0) {
                    shannonEntropy -= binValue * std::log10(binValue);
                }
            }
        }

        descriptor.contents[5 * histogramIndex + 0] = centralMoment_11;
        descriptor.contents[5 * histogramIndex + 1] = centralMoment_21;
        descriptor.contents[5 * histogramIndex + 2] = centralMoment_12;
        descriptor.contents[5 * histogramIndex + 3] = centralMoment_22;
        descriptor.contents[5 * histogramIndex + 4] = shannonEntropy;
    }

    return descriptor;
}

ShapeDescriptor::cpu::array<ShapeDescriptor::RoPSDescriptor> ShapeDescriptor::generateRoPSDescriptors(
        ShapeDescriptor::cpu::Mesh mesh,
        ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> descriptorOrigins,
        float supportRadius,
        float numPointSamplesPerUnitArea,
        uint64_t randomSeed,
        uint32_t pointSampleCountLimit,
        ShapeDescriptor::RoPSExecutionTimes* executionTimes) {

    ShapeDescriptor::cpu::array<ShapeDescriptor::RoPSDescriptor> outputDescriptors(descriptorOrigins.length);
    double meshArea = 0;
    std::vector<std::array<ShapeDescriptor::cpu::float3, 3>> localReferenceFrames = computeLocalReferenceFrames(mesh, descriptorOrigins, supportRadius, meshArea);

    uint32_t meshSampleCount = uint32_t(meshArea * numPointSamplesPerUnitArea);
    meshSampleCount = std::min<uint32_t>(meshSampleCount, pointSampleCountLimit);
    ShapeDescriptor::cpu::PointCloud cloud = ShapeDescriptor::sampleMesh(mesh, meshSampleCount, randomSeed);

    // Can happen if a mesh pretends to be a point cloud
    if(meshArea == 0) {
        for(uint32_t descriptorIndex = 0; descriptorIndex < descriptorOrigins.length; descriptorIndex++) {
            for(float& bin : outputDescriptors.content[descriptorIndex].contents) {
                bin = 0;
            }
        }
        return outputDescriptors;
    }

    //#pragma omp parallel for
    for(uint32_t descriptorIndex = 0; descriptorIndex < descriptorOrigins.length; descriptorIndex++) {
        outputDescriptors[descriptorIndex] = computeRoPSDescriptor(cloud,
                                                                   descriptorOrigins[descriptorIndex],
                                                                   localReferenceFrames.at(descriptorIndex),
                                                                   supportRadius);
        for(uint32_t binIndex = 0; binIndex < 3 * 3 * ROPS_NUM_ROTATIONS * ROPS_HISTOGRAM_BINS; binIndex++) {
            float bin = outputDescriptors.content[descriptorIndex].contents[binIndex];
            if(std::isnan(bin)) {
                throw std::runtime_error("Found a NaN!");
            }
        }
    }

    ShapeDescriptor::free(cloud);



    return outputDescriptors;
}