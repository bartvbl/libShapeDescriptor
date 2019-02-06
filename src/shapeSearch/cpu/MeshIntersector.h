#pragma once

#include "shapeSearch/common/geom.hpp"
#include "hostMesh.h"

#include <glm/detail/type_mat4x4.hpp>
#include <vector>

struct VertexAtZeroCrossing {
	glm::vec4 edgeVertexTop;
	glm::vec4 edgeVertexBottom;
};

struct IntersectionLineSegment {
	VertexAtZeroCrossing endVertex0;
	VertexAtZeroCrossing endVertex1;
};

struct IntersectionCluster {
    VertexAtZeroCrossing clusterStart;
    VertexAtZeroCrossing clusterEnd;

    std::vector<IntersectionLineSegment> contents;
};

struct IntersectResult2D {
	int count;
};

void computePlaneIntersections(glm::vec4 vertices[], unsigned int triangleCount, glm::mat4 transformations[], std::vector<IntersectionLineSegment> intersections[], int planeStepCount);

glm::mat4 generateAlignmentTransformation(const float3_cpu &origin, const float3_cpu &normal, const float &planeAngleRadians);
std::vector<IntersectionCluster> linkIntersectionEdges(std::vector<IntersectionLineSegment> intersectingEdges);
std::vector<IntersectionLineSegment> intersectPlane(HostMesh mesh, float3_cpu origin, float3_cpu normal, float planeAngle);