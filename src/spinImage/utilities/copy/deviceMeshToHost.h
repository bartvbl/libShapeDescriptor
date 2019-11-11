#pragma once

#include <spinImage/cpu/types/Mesh.h>
#include <spinImage/gpu/types/Mesh.h>

namespace SpinImage {
    namespace copy{
        cpu::Mesh deviceMeshToHost(gpu::Mesh deviceMesh);
    }
}