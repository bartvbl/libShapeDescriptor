#pragma once

#include "shapeSearch/cpu/types/HostMesh.h"

#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>

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

namespace SpinImage {
	namespace cpu {
		void computeMeshPlaneIntersections(
				glm::vec4 vertices[],
				unsigned int triangleCount,
				glm::mat4 transformations[],
				std::vector<IntersectionLineSegment> intersections[],
				const int planeStepCount);
	}
}