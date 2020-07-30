#pragma once

#include "float3.h"
#include "float2.h"


namespace SpinImage {
	namespace cpu {
        struct Mesh {
            float3* vertices;
            float3* normals;

            unsigned int* indices;

            size_t vertexCount;
            size_t indexCount;

            Mesh() {
                vertices = nullptr;
                normals = nullptr;
                indices = nullptr;
                vertexCount = 0;
                indexCount = 0;
            }

            Mesh(size_t vertCount, size_t numIndices) {
                vertices = new float3[vertCount];
                normals = new float3[vertCount];

                indices = new unsigned int[numIndices];
                indexCount = numIndices;
                vertexCount = vertCount;
            }
        };

        void freeMesh(SpinImage::cpu::Mesh &mesh);
    }
}

