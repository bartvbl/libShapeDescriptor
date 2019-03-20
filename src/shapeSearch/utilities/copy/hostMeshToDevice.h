#pragma once

#include <shapeSearch/cpu/types/HostMesh.h>
#include <shapeSearch/gpu/types/DeviceMesh.h>

namespace SpinImage::copy {
    DeviceMesh hostMeshToDevice(HostMesh hostMesh);
}