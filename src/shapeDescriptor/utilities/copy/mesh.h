#pragma once

#include <shapeDescriptor/cpu/types/Mesh.h>
#include <shapeDescriptor/gpu/types/Mesh.h>

namespace ShapeDescriptor {
    namespace copy{
        cpu::Mesh deviceMeshToHost(gpu::Mesh deviceMesh);

        gpu::Mesh hostMeshToDevice(cpu::Mesh hostMesh);
    }
}