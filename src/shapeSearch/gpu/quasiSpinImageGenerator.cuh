#pragma once

#include <shapeSearch/libraryBuildSettings.h>
#include <shapeSearch/gpu/types/DeviceMesh.h>

#include "cuda_runtime.h"

array<newSpinImagePixelType> createDescriptorsNewstyle(DeviceMesh device_mesh, cudaDeviceProp device_information, float spinImageWidth);