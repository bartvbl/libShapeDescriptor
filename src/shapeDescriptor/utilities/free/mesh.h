#pragma once

#include <shapeDescriptor/cpu/types/Mesh.h>
#include <shapeDescriptor/gpu/types/Mesh.h>

namespace ShapeDescriptor {
    namespace free {
        void mesh(ShapeDescriptor::cpu::Mesh meshToFree);
        void mesh(ShapeDescriptor::gpu::Mesh meshToFree);
    }
}