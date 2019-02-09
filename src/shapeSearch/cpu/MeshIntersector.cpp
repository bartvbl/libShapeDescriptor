#include <vector>
#include <iostream>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/transform.hpp>

#include "MeshIntersector.h"

std::vector<IntersectionCluster> linkEdgeChains(std::vector<IntersectionLineSegment> vector);
std::vector<IntersectionLineSegment> findPlaneIntersections(const HostMesh &mesh, const glm::mat4 &alignmentTransformation);
glm::mat4 generateAlignmentTransformation(const float3_cpu &origin, const float3_cpu &normal, const float &planeAngle);
float3_cpu vec4tofloat3(glm::vec4 in);
void assignEdge(VertexAtZeroCrossing &edge, glm::vec4 vertex0, glm::vec4 vertex1);

enum ClusterSide {
    START, END
};

std::vector<IntersectionCluster> linkIntersectionEdges(std::vector<IntersectionLineSegment> intersectingEdges) {
    std::vector<IntersectionCluster> clusteredEdges = linkEdgeChains(intersectingEdges);
    return clusteredEdges;
}

glm::mat4 generateAlignmentTransformation(const float3_cpu &origin, const float3_cpu &normal, const float &planeAngleRadians) {
    const glm::vec3 targetCoordinateSystemX(1, 0, 0);
    const glm::vec3 targetCoordinateSystemY(0, -1, 0);
    const glm::vec3 targetCoordinateSystemZ(0, 0, 1);

    // COLUMN major order
    float targetTransform[16] = {
            targetCoordinateSystemX.x, targetCoordinateSystemY.x, targetCoordinateSystemZ.x, 0,
            targetCoordinateSystemX.y, targetCoordinateSystemY.y, targetCoordinateSystemZ.y, 0,
            targetCoordinateSystemX.z, targetCoordinateSystemY.z, targetCoordinateSystemZ.z, 0,
            0                        , 0                        , 0                        , 1
    };
    glm::mat4 targetCoordinateSystem = glm::make_mat4(targetTransform);

    glm::vec3 glmNormal = glm::normalize(glm::vec3(normal.x, normal.y, normal.z));
    const glm::vec3 zAxis(0, 0, 1);
    glm::vec3 basePlaneDirection = glm::normalize(glm::cross(zAxis, glmNormal));
    glm::vec3 sourceXAxis = glm::normalize(glm::cross(basePlaneDirection, glmNormal));

    glm::vec3 sourceCoordinateSystemX = basePlaneDirection;
    glm::vec3 sourceCoordinateSystemY = glmNormal;
    glm::vec3 sourceCoordinateSystemZ = sourceXAxis;

    // COLUMN major order
    float sourceTransform[16] = {
            sourceCoordinateSystemX.x, sourceCoordinateSystemY.x, sourceCoordinateSystemZ.x, 0,
            sourceCoordinateSystemX.y, sourceCoordinateSystemY.y, sourceCoordinateSystemZ.y, 0,
            sourceCoordinateSystemX.z, sourceCoordinateSystemY.z, sourceCoordinateSystemZ.z, 0,
            0                        , 0                        , 0                        , 1
    };
    glm::mat4 sourceCoordinateSystem = glm::make_mat4(sourceTransform);

    glm::mat4 alignmentTransformation =
            glm::rotate(planeAngleRadians, glm::vec3(0, 1, 0)) *
            targetCoordinateSystem *
            glm::inverse(sourceCoordinateSystem) * glm::translate(glm::vec3(-origin.x, -origin.y, -origin.z));

    return alignmentTransformation;
}

void computePlaneIntersections(glm::vec4 vertices[], unsigned int triangleCount, glm::mat4 transformations[], std::vector<IntersectionLineSegment> intersections[], const int planeStepCount) {

#pragma omp for
    for(unsigned int triangleIndex = 0; triangleIndex < triangleCount; triangleIndex++) {

        // THIS ASSUMES WE DUPLICATED VERTICES, ELIMINATING THE NEED FOR AN INDEX BUFFER!!!
        glm::vec4 vertex0 = vertices[3 * triangleIndex + 0];
        glm::vec4 vertex1 = vertices[3 * triangleIndex + 1];
        glm::vec4 vertex2 = vertices[3 * triangleIndex + 2];

		for(int planeStep = 0; planeStep < planeStepCount; planeStep++) {

		    glm::mat4 alignmentTransformation = transformations[planeStep];

            glm::vec4 transformedVertex0 = alignmentTransformation * vertex0;
            glm::vec4 transformedVertex1 = alignmentTransformation * vertex1;
            glm::vec4 transformedVertex2 = alignmentTransformation * vertex2;

            bool firstIntersectingEdgeFound = false;
            bool secondIntersectionEdgeFound = false;

            IntersectionLineSegment intersection{};

            // Cheap way to check whether the triangle edge contains a zero crossing
            if (copysign(1, transformedVertex0.z * transformedVertex1.z) < 0) {
                firstIntersectingEdgeFound = true;
                assignEdge(intersection.endVertex0, transformedVertex0, transformedVertex1);
            }

            if (copysign(1, transformedVertex1.z * transformedVertex2.z) < 0) {
                if (firstIntersectingEdgeFound) {
                    assignEdge(intersection.endVertex1, transformedVertex1, transformedVertex2);
                    secondIntersectionEdgeFound = true;
                } else {
                    assignEdge(intersection.endVertex0, transformedVertex1, transformedVertex2);
                }
                firstIntersectingEdgeFound = true;
            }

            if (copysign(1, transformedVertex2.z * transformedVertex0.z) < 0) {
                if (firstIntersectingEdgeFound) {
                    assignEdge(intersection.endVertex1, transformedVertex2, transformedVertex0);
                    secondIntersectionEdgeFound = true;
                } else {
                    std::cerr << "Only one of two zero crossings found (on line " << __LINE__ << ")" << std::endl;
                    std::cerr << "Triangle vertices (untransformed):" << std::endl;
                    std::cerr << "\t" << vertex0.x << ", " << vertex0.y << ", " << vertex0.z << std::endl;
                    std::cerr << "\t" << vertex1.x << ", " << vertex1.y << ", " << vertex1.z << std::endl;
                    std::cerr << "\t" << vertex2.x << ", " << vertex2.y << ", " << vertex2.z << std::endl;
                    std::cerr << "Triangle vertices (transformed):" << std::endl;
                    std::cerr << "\t" << transformedVertex0.x << ", " << transformedVertex0.y << ", "
                              << transformedVertex0.z << std::endl;
                    std::cerr << "\t" << transformedVertex1.x << ", " << transformedVertex1.y << ", "
                              << transformedVertex1.z << std::endl;
                    std::cerr << "\t" << transformedVertex2.x << ", " << transformedVertex2.y << ", "
                              << transformedVertex2.z << std::endl;
                    std::cerr << "Detected edge combinations:" << std::endl;
                    std::cerr << "vertex 0 and 1:" << transformedVertex0.z * transformedVertex1.z << " -> "
                              << (transformedVertex0.z * transformedVertex1.z < 0) << std::endl;
                    std::cerr << "vertex 1 and 2:" << transformedVertex1.z * transformedVertex2.z << " -> "
                              << (transformedVertex1.z * transformedVertex2.z < 0) << std::endl;
                    std::cerr << "vertex 2 and 0:" << transformedVertex2.z * transformedVertex0.z << " -> "
                              << (transformedVertex2.z * transformedVertex0.z < 0) << std::endl;

                    assert(false);
                }
            }

            // We now have found the two edges of the triangle which intersect the origin.
            if (firstIntersectingEdgeFound && secondIntersectionEdgeFound) {
#pragma omp critical
                {
                    intersections[planeStep].push_back(intersection);
                }
            } else if (firstIntersectingEdgeFound /*&& !secondIntersectionEdgeFound */) {
                std::cerr << "Only one of two zero crossings found (on line " << __LINE__ << ")" << std::endl;
                assert(false);
            }
        }
	}
}

bool isEdgeEqual(VertexAtZeroCrossing edge1, VertexAtZeroCrossing edge2) {
    return (edge1.edgeVertexTop == edge2.edgeVertexTop) && (edge1.edgeVertexBottom == edge2.edgeVertexBottom);
}

void mergeClusters(std::vector<IntersectionCluster> &clusters, unsigned int clusterIndex1, unsigned int clusterIndex2, ClusterSide cluster1Side, ClusterSide cluster2Side) {
    IntersectionCluster cluster1 = clusters.at(clusterIndex1);
    IntersectionCluster cluster2 = clusters.at(clusterIndex2);

    // There are two clusters here each with a start and end. The clusterSide parameters of this function tell which ends of each cluster should be connected together
    // That means the other ends of each cluster become the new start and end of the merged cluster.
    // We're merging cluster 2 into cluster 1 here.
    clusters.at(clusterIndex1).clusterStart = cluster1Side == START ? cluster1.clusterEnd : cluster1.clusterStart;
    clusters.at(clusterIndex1).clusterEnd   = cluster2Side == START ? cluster2.clusterEnd : cluster2.clusterStart;

    clusters.at(clusterIndex1).contents.insert(clusters.at(clusterIndex1).contents.end(), cluster2.contents.begin(), cluster2.contents.end());
    clusters.erase(clusters.begin() + clusterIndex2);
}

std::vector<IntersectionCluster> linkEdgeChains(std::vector<IntersectionLineSegment> vector) {

    std::vector<IntersectionCluster> clusters;

    clusters.reserve(vector.size());

    // Initialise cluster array by creating a new cluster per edge in the input array
    for(unsigned int i = 0; i < vector.size(); i++) {
        IntersectionCluster cluster;
        IntersectionLineSegment segment = vector.at(i);
        cluster.contents.push_back(segment);
        cluster.clusterStart = segment.endVertex0;
        cluster.clusterEnd = segment.endVertex1;
        clusters.push_back(cluster);
    }

    // Most inefficient clustering algorithm ever: loop through list, connect points that should be connected
    // until no more connections can be made.

    bool changesOccurred = true;

    while(changesOccurred) {
        changesOccurred = false;

        for(unsigned int clusterIndex = 0; clusterIndex < clusters.size(); clusterIndex++) {
            IntersectionCluster currentCluster = clusters.at(clusterIndex);

            // Comparisons are cummutative, so we can start further on in the list
            for(unsigned int otherClusterIndex = clusterIndex + 1; otherClusterIndex < clusters.size(); otherClusterIndex++) {
                if(clusterIndex == otherClusterIndex) {
                    continue;
                }

                IntersectionCluster otherClusterStart = clusters.at(otherClusterIndex);


                if(isEdgeEqual(currentCluster.clusterStart, otherClusterStart.clusterStart)) {
                    mergeClusters(clusters, clusterIndex, otherClusterIndex, START, START);
                    otherClusterIndex--;
                    changesOccurred = true;
                }
                else if(isEdgeEqual(currentCluster.clusterEnd, otherClusterStart.clusterStart)) {
                    mergeClusters(clusters, clusterIndex, otherClusterIndex, END, START);
                    changesOccurred = true;
                    otherClusterIndex--;
                }
                else if(isEdgeEqual(currentCluster.clusterEnd, otherClusterStart.clusterEnd)) {
                    mergeClusters(clusters, clusterIndex, otherClusterIndex, END, END);
                    changesOccurred = true;
                    otherClusterIndex--;
                }
                else if(isEdgeEqual(currentCluster.clusterStart, otherClusterStart.clusterEnd)) {
                    mergeClusters(clusters, clusterIndex, otherClusterIndex, START, END);
                    changesOccurred = true;
                    otherClusterIndex--;
                }
            }
        }
    }

    return clusters;
}

void assignEdge(VertexAtZeroCrossing &edge, glm::vec4 vertex0, glm::vec4 vertex1) {
    edge.edgeVertexTop = vertex0.z > vertex1.z ? vertex0 : vertex1;
    edge.edgeVertexBottom = vertex0.z > vertex1.z ? vertex1 : vertex0;
}