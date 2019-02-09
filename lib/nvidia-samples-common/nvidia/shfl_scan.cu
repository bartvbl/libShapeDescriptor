/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

 /*

// Shuffle intrinsics CUDA Sample
// This sample demonstrates the use of the shuffle intrinsic
// First, a simple example of a prefix sum using the shuffle to
// perform a scan operation is provided.
// Secondly, a more involved example of computing an integral image
// using the shuffle intrinsic is provided, where the shuffle
// scan operation and shuffle xor operations are used

#include "shfl_scan.cuh"
#include <stdio.h>

#include <cuda_runtime.h>

#include "helper_functions.h"
#include "helper_cuda.h"
#include "shfl_integral_image.cuh"
#include <device_launch_parameters.h>
#include "helper_math.h"
#include "sharedmem.cuh"

// Scan using shfl - takes log2(n) steps
// This function demonstrates basic use of the shuffle intrinsic, __shfl_up,
// to perform a scan operation across a block.
// First, it performs a scan (prefix sum in this case) inside a warp
// Then to continue the scan operation across the block,
// each warp's sum is placed into shared memory.  A single warp
// then performs a shuffle scan on that shared memory.  The results
// are then uniformly added to each warp's threads.
// This pyramid type approach is continued by placing each block's
// final sum in global memory and prefix summing that via another kernel call, then
// uniformly adding across the input data via the uniform_add<<<>>> kernel.

template<class TYPE>
__global__ void shfl_scan(TYPE *src, TYPE* dest, int dataLength, int width, TYPE *partial_sums=NULL)
{
	SharedMemory<TYPE> sharedSums;
	TYPE* sums = sharedSums.getPointer();
    int id = ((blockIdx.x * blockDim.x) + threadIdx.x);
    int lane_id = id % warpSize;
    // determine a warp_id within a block
    int warp_id = threadIdx.x / warpSize;

	if(id >= dataLength)
	{
		return;
	}

    // Below is the basic structure of using a shfl instruction
    // for a scan.
    // Record "value" as a variable - we accumulate it along the way
	TYPE value = src[id];

    // Now accumulate in log steps up the chain
    // compute sums, with another thread's value who is
    // distance delta away (i).  Note
    // those threads where the thread 'i' away would have
    // been out of bounds of the warp are unaffected.  This
    // creates the scan sum.
#pragma unroll

    for (int i = 1; i < width; i *= 2)
    {
		TYPE n = __shfl_up(value, i, width);

        if (lane_id >= i) value += n;
    }

    // value now holds the scan value for the individual thread
    // next sum the largest values for each warp

    // write the sum of the warp to smem
    if (threadIdx.x % warpSize == warpSize-1)
    {
		sums[warp_id] = value;
    }

    __syncthreads();

    //
    // scan sum the warp sums
    // the same shfl scan operation, but performed on warp sums
    //
    if (warp_id == 0 && lane_id < (blockDim.x / warpSize))
    {
		TYPE warp_sum = sums[lane_id];

        for (int i = 1; i < width; i *= 2)
        {
			TYPE n = __shfl_up(warp_sum, i, width);

            if (lane_id >= i) warp_sum += n;
        }

        sums[lane_id] = warp_sum;
    }

    __syncthreads();

    // perform a uniform add across warps in the block
    // read neighbouring warp's sum and add it to threads value
	TYPE blockSum = 0;

    if (warp_id > 0)
    {
        blockSum = sums[warp_id-1];
    }

    value += blockSum;

    // Now write out our result
	dest[id] = value;

    // last thread has sum, write write out the block's sum
    if (partial_sums != NULL && threadIdx.x == blockDim.x-1)
    {
        partial_sums[blockIdx.x] = value;
    }
}

// Uniform add: add partial sums array
template<typename TYPE>
__global__ void uniform_add(TYPE *input_data, TYPE *output_data, TYPE *partial_sums, int len)
{
    __shared__ TYPE buf;
    int id = ((blockIdx.x * blockDim.x) + threadIdx.x);

    if (id >= len) return;

    if (threadIdx.x == 0)
    {
        buf = partial_sums[blockIdx.x];
    }

    __syncthreads();
    output_data[id] = input_data[id] + buf;
	//printf("output\t%i\t%i\n", id, (input_data[id] + buf));
}

static unsigned int iDivUp(unsigned int dividend, unsigned int divisor)
{
    return ((dividend % divisor) == 0) ?
           (dividend / divisor) :
           (dividend / divisor + 1);
}

template<typename TYPE>
void shuffle_prefix_scan(TYPE* device_input, TYPE* device_output, int elementCount)
{
    if(device_input != device_output)
    {
		checkCudaErrors(cudaMemcpy(device_output, device_input, sizeof(TYPE) * elementCount, cudaMemcpyDeviceToDevice));
    }
	TYPE *d_partial_sums;

    int blockSize = 256;
    int gridSize = iDivUp(elementCount, blockSize);
    int nWarps = blockSize/32;
    int shmem_sz = nWarps * sizeof(int);
    int n_partialSums = std::max(elementCount / blockSize,1);
    int partial_sz = n_partialSums*sizeof(int);

    printf("Scan summation for %d elements, %d partial sums\n", elementCount, elementCount / blockSize);

    int p_blockSize = min(n_partialSums, blockSize);
    int p_gridSize = iDivUp(n_partialSums, p_blockSize);
    printf("Partial summing %d elements with %d blocks of size %d\n", n_partialSums, p_gridSize, p_blockSize);

    checkCudaErrors(cudaMalloc(&d_partial_sums, partial_sz));
    checkCudaErrors(cudaMemset(d_partial_sums, 0, partial_sz));

    shfl_scan<TYPE> <<<std::max(gridSize,1),blockSize, shmem_sz>>>(device_output, device_output, elementCount, 32, d_partial_sums);
    if(elementCount / blockSize > 0)
    {
		shfl_scan<TYPE> <<<p_gridSize,p_blockSize, shmem_sz>>>(d_partial_sums, d_partial_sums, n_partialSums, 32);
		uniform_add<TYPE> <<<std::max(gridSize-1,1), blockSize>>>(device_output + blockSize, device_output + blockSize, d_partial_sums, elementCount - blockSize);
    }

	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaFree(d_partial_sums));
}

void shuffle_prefix_scan_float(float * device_input, float * device_output, int elementCount)
{
	shuffle_prefix_scan<float>(device_input, device_output, elementCount);
}

void shuffle_prefix_scan_int(int * device_input, int * device_output, int elementCount)
{
	shuffle_prefix_scan<int>(device_input, device_output, elementCount);
}

void shuffle_prefix_scan_uint(unsigned int * device_input, unsigned int * device_output, int elementCount)
{
	shuffle_prefix_scan<unsigned int>(device_input, device_output, elementCount);
}*/