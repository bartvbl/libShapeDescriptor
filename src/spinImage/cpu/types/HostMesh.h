#pragma once

#include "spinImage/common/types/array.h"
#include "float3_cpu.h"
#include "float2_cpu.h"


struct HostMesh {
	float3_cpu* vertices;
	float3_cpu* normals;

	unsigned int* indices;

	size_t vertexCount;
	size_t indexCount;

	HostMesh() {
		vertices = nullptr;
		normals = nullptr;
		indices = nullptr;
		vertexCount = 0;
		indexCount = 0;
	}

	HostMesh(size_t vertCount, size_t numIndices) {
		vertices = new float3_cpu[vertCount];
        normals = new float3_cpu[vertCount];

		indices = new unsigned int[numIndices];
		indexCount = numIndices;
		vertexCount = vertCount;
	}
};

namespace SpinImage {
	namespace cpu {
		void freeHostMesh(HostMesh &mesh);
	}
}

