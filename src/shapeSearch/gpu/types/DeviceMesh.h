#pragma once
#include "shapeSearch/common/types/array.h"
#include <shapeSearch/common/types/MeshFormat.h>
#include <host_defines.h>

struct DeviceMesh {
    float* vertices_x;
    float* vertices_y;
    float* vertices_z;

    float* normals_x;
    float* normals_y;
    float* normals_z;

    size_t vertexCount;
    size_t indexCount;

	__host__ __device__ DeviceMesh() {
		vertexCount = 0;
		indexCount = 0;
	}
};

