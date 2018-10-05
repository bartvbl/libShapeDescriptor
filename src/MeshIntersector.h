#pragma once

#include "geom.hpp"
#include "Mesh.h"

#include <glm/detail/type_mat4x4.hpp>
#include <glm/gtx/vector_angle.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <vector>

typedef struct VertexAtZeroCrossing {
	glm::vec4 edgeVertexTop;
	glm::vec4 edgeVertexBottom;
} VertexAtZeroCrossing;

typedef struct IntersectionLineSegment {
	VertexAtZeroCrossing endVertex0;
	VertexAtZeroCrossing endVertex1;
};

typedef struct IntersectionCluster {
    VertexAtZeroCrossing clusterStart;
    VertexAtZeroCrossing clusterEnd;

    std::vector<IntersectionLineSegment> contents;
};

typedef struct IntersectResult2D {
	int count;
} IntersectResult2D;

void computePlaneIntersections(glm::vec4 vertices[], unsigned int triangleCount, glm::mat4 transformations[], std::vector<IntersectionLineSegment> intersections[], int planeStepCount);

glm::mat4 generateAlignmentTransformation(const float3 &origin, const float3 &normal, const float &planeAngleRadians);
std::vector<IntersectionCluster> linkIntersectionEdges(std::vector<IntersectionLineSegment> intersectingEdges);
std::vector<IntersectionLineSegment> intersectPlane(Mesh mesh, float3 origin, float3 normal, float planeAngle);