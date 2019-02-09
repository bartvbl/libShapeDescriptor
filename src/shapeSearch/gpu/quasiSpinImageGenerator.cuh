#pragma once

#include <shapeSearch/common/types/VertexDescriptors.h>
#include <shapeSearch/gpu/types/DeviceMesh.h>

#include "cuda_runtime.h"

VertexDescriptors createDescriptorsNewstyle(DeviceMesh device_mesh, cudaDeviceProp device_information);