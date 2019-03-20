#pragma once

#include <shapeSearch/cpu/types/HostMesh.h>
#include <shapeSearch/gpu/types/DeviceMesh.h>

DeviceMesh copyMeshToGPU(HostMesh hostMesh);