#include <cuda_runtime.h>

__global__ void prescan(float *input, float *output, int n);
__global__ void prescan_integer(int *input, int *output, int n);