#pragma once
#include "shapeSearch/cpu/arrayTypes.hpp"
#include "geom.hpp"

enum MeshFormat {
	VERTICES,
	VERTICES_TEXCOORDS,
	VERTICES_NORMALS,
	VERTICES_TEXCOORDS_NORMALS
};

typedef struct HostMesh {
	float3_cpu* vertices;
	float3_cpu* normals;
	float2_cpu* textureCoordinates;

	MeshFormat dataFormat;

	unsigned int* indices;

	size_t vertexCount;
	size_t indexCount;

	float3_cpu boundingBoxMin;
	float3_cpu boundingBoxMax;

	HostMesh() {
		vertices = nullptr;
		normals = nullptr;
		textureCoordinates = nullptr;
		indices = nullptr;
		dataFormat = VERTICES;
		vertexCount = 0;
		indexCount = 0;
		boundingBoxMin = {0, 0, 0};
		boundingBoxMax = {0, 0, 0};
	}

	HostMesh(size_t vertCount, size_t numIndices, MeshFormat format) {
		vertices = new float3_cpu[vertCount];

		if(format == VERTICES_NORMALS || format == VERTICES_TEXCOORDS_NORMALS) {
			normals = new float3_cpu[vertCount];
		} else {
			normals = nullptr;
		}

		if(format == VERTICES_TEXCOORDS || format == VERTICES_TEXCOORDS_NORMALS) {
			textureCoordinates = new float2_cpu[vertCount];
		} else {
			textureCoordinates = nullptr;
		}

		indices = new unsigned int[numIndices];
		indexCount = numIndices;
		vertexCount = vertCount;

		dataFormat = format;

		boundingBoxMin = {0, 0, 0};
		boundingBoxMax = {5, 5, 5};
	}

	void deleteMesh() {
		delete[] vertices;
		delete[] normals;
		delete[] indices;
		delete[] textureCoordinates;
	}


} HostMesh;

