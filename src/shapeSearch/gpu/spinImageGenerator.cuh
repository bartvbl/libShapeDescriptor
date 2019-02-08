#pragma once

#include <shapeSearch/common/types/VertexDescriptors.h>
#include <shapeSearch/common/types/OutputImageSettings.h>
#include <shapeSearch/gpu/types/DeviceMesh.h>

VertexDescriptors createClassicDescriptors(DeviceMesh device_mesh, cudaDeviceProp device_information, size_t sampleCount);