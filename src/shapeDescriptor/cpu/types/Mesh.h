#pragma once

#include <shapeDescriptor/cpu/types/float3.h>
#include <shapeDescriptor/cpu/types/float2.h>


namespace ShapeDescriptor {
	namespace cpu {
        struct Mesh {
            float3* vertices = nullptr;
            float3* normals = nullptr;

            size_t vertexCount = 0;

            Mesh() {}

            Mesh(size_t vertCount) {
                vertices = new float3[vertCount];
                normals = new float3[vertCount];
                vertexCount = vertCount;
            }

            ShapeDescriptor::cpu::Mesh clone() const;
        };
    }
}

