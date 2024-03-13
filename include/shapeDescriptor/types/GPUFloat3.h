#pragma once

#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
#include <cuda_runtime_api.h>
#else
#include <shapeDescriptor/types/float3.h>
typedef ShapeDescriptor::cpu::float3 float3;
#endif