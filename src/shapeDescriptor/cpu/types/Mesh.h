#pragma once

#include <shapeDescriptor/cpu/types/float3.h>
#include <shapeDescriptor/cpu/types/float2.h>
#include "uchar4.h"


namespace ShapeDescriptor {
	namespace cpu {
        struct Mesh {
            ShapeDescriptor::cpu::float3* vertices = nullptr;
            ShapeDescriptor::cpu::float3* normals = nullptr;
            ShapeDescriptor::cpu::uchar4* vertexColours = nullptr;

            size_t vertexCount = 0;

            Mesh() = default;

            Mesh(size_t vertCount) {
                vertices = new ShapeDescriptor::cpu::float3[vertCount];
                normals = new ShapeDescriptor::cpu::float3[vertCount];
                vertexColours = new ShapeDescriptor::cpu::uchar4[vertexCount];
                vertexCount = vertCount;
            }

            ShapeDescriptor::cpu::Mesh clone() const;
        };
    }
}

