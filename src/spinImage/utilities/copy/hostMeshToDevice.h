#pragma once

#include <spinImage/cpu/types/HostMesh.h>
#include <spinImage/gpu/types/DeviceMesh.h>

namespace SpinImage {
    namespace copy{
        DeviceMesh hostMeshToDevice(HostMesh hostMesh);
    }
}