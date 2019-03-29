#pragma once

#ifndef __CUDACC__
#define __host__
#define __device__
#endif

__device__ float computeSpinImagePairCorrelationGPU(
        spinImagePixelType* descriptors,
        spinImagePixelType* otherDescriptors,
        size_t spinImageIndex,
        size_t otherImageIndex,
        float averageX, float averageY);

__host__ float computeSpinImagePairCorrelationCPU(
        spinImagePixelType* descriptors,
        spinImagePixelType* otherDescriptors,
        size_t spinImageIndex,
        size_t otherImageIndex,
        float averageX, float averageY);