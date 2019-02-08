#pragma once

#include <shapeSearch/common/types/vertexDescriptors.h>
#include <shapeSearch/common/types/outputImageSettings.h>
#include <shapeSearch/gpu/deviceMesh.h>

VertexDescriptors createClassicDescriptors(DeviceMesh device_mesh, cudaDeviceProp device_information, OutputImageSettings imageSettings, size_t sampleCount);