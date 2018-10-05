#pragma once
#include "shapeSearch/geom.hpp"
#include "shapeSearch/arrayTypes.hpp"

enum MeshFormat {
	VERTICES,
	VERTICES_TEXCOORDS,
	VERTICES_NORMALS,
	VERTICES_TEXCOORDS_NORMALS
};

typedef struct Mesh {
	float3* vertices;
	float3* normals;
	float2* textureCoordinates;

	MeshFormat dataFormat;

	unsigned int* indices;

	size_t vertexCount;
	size_t indexCount;

	float3 boundingBoxMin;
	float3 boundingBoxMax;

	Mesh() {
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

	Mesh(size_t vertCount, size_t numIndices, MeshFormat format) {
		vertices = new float3[vertCount];

		if(format == VERTICES_NORMALS || format == VERTICES_TEXCOORDS_NORMALS) {
			normals = new float3[vertCount];
		} else {
			normals = nullptr;
		}

		if(format == VERTICES_TEXCOORDS || format == VERTICES_TEXCOORDS_NORMALS) {
			textureCoordinates = new float2[vertCount];
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
} Mesh;