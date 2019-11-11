#pragma once

#include <spinImage/cpu/types/Mesh.h>
#include <spinImage/gpu/types/Mesh.h>

namespace SpinImage {
    namespace copy{
        gpu::Mesh hostMeshToDevice(cpu::Mesh hostMesh);
    }
}