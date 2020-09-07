#pragma once

#include <shapeDescriptor/cpu/types/float3.h>
#include <shapeDescriptor/cpu/types/float2.h>


namespace ShapeDescriptor {
	namespace cpu {
        struct Mesh {
            float3* vertices;
            float3* normals;

            size_t vertexCount;

            Mesh() {
                vertices = nullptr;
                normals = nullptr;
                vertexCount = 0;
            }

            Mesh(size_t vertCount) {
                vertices = new float3[vertCount];
                normals = new float3[vertCount];
                vertexCount = vertCount;
            }
        };

        void freeMesh(ShapeDescriptor::cpu::Mesh &mesh);
    }
}

