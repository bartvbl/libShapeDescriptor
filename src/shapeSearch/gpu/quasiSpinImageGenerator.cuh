#pragma once

#include <shapeSearch/common/types/VertexDescriptors.h>
#include <shapeSearch/gpu/types/DeviceMesh.h>
#include <shapeSearch/common/types/OutputImageSettings.h>

VertexDescriptors createDescriptorsNewstyle(DeviceMesh device_mesh, cudaDeviceProp device_information);