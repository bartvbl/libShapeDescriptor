#pragma once

#include "shapeSearch/common/types/array.h"
#include "float3_cpu.h"
#include "float2_cpu.h"


struct HostMesh {
	float3_cpu* vertices;
	float3_cpu* normals;

	unsigned int* indices;

	size_t vertexCount;
	size_t indexCount;

	float3_cpu boundingBoxMin;
	float3_cpu boundingBoxMax;

	HostMesh() {
		vertices = nullptr;
		normals = nullptr;
		indices = nullptr;
		vertexCount = 0;
		indexCount = 0;
		boundingBoxMin = {0, 0, 0};
		boundingBoxMax = {0, 0, 0};
	}

	HostMesh(size_t vertCount, size_t numIndices) {
		vertices = new float3_cpu[vertCount];
        normals = new float3_cpu[vertCount];

		indices = new unsigned int[numIndices];
		indexCount = numIndices;
		vertexCount = vertCount;

		boundingBoxMin = {0, 0, 0};
		boundingBoxMax = {5, 5, 5};
	}
};

namespace SpinImage {
	namespace cpu {
		void freeHostMesh(HostMesh &mesh);
	}
}

