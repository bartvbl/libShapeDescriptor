#pragma once

#include <spinImage/libraryBuildSettings.h>

#ifndef __CUDACC__
#define __host__
#define __device__
#endif

__device__ __inline__ int compareQuasiSpinImagePairGPU(
        quasiSpinImagePixelType* needleImages,
        size_t needleImageIndex,
        quasiSpinImagePixelType* haystackImages,
        size_t haystackImageIndex);