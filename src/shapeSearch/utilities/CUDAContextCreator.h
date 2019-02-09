#pragma once

#include "cuda_runtime.h"

cudaDeviceProp createCUDAContext(int forceGPU);
void printGPUProperties(unsigned int deviceIndex);