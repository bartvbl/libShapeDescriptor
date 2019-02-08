#pragma once
#include "shapeSearch/common/types/arrayTypes.hpp"
#include "shapeSearch/common/geom.hpp"
#include <shapeSearch/common/meshFormat.h>

struct DeviceMesh {
    float* vertices_x;
    float* vertices_y;
    float* vertices_z;

    float* normals_x;
    float* normals_y;
    float* normals_z;

    size_t vertexCount;
    size_t indexCount;

	DeviceMesh() {
		vertexCount = 0;
		indexCount = 0;
	}
};

