#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "prefix_scan.cuh"

//const int BLOCK_SIZE = 32;

// From: https://gist.github.com/wh5a/4500706
/*__global__ void prescan(float * input, float * output, int len) {
	// Load a segment of the input vector into shared memory
	__shared__ float scan_array[BLOCK_SIZE << 1];
	unsigned int t = threadIdx.x, start = 2 * blockIdx.x * BLOCK_SIZE;
	if (start + t < len)
		scan_array[t] = input[start + t];
	else
		scan_array[t] = 0;
	if (start + BLOCK_SIZE + t < len)
		scan_array[BLOCK_SIZE + t] = input[start + BLOCK_SIZE + t];
	else
		scan_array[BLOCK_SIZE + t] = 0;
	__syncthreads();

	// Reduction
	int stride;
	for (stride = 1; stride <= BLOCK_SIZE; stride <<= 1) {
		int index = (t + 1) * stride * 2 - 1;
		if (index < 2 * BLOCK_SIZE)
			scan_array[index] += scan_array[index - stride];
		__syncthreads();
	}

	// Post reduction
	for (stride = BLOCK_SIZE >> 1; stride; stride >>= 1) {
		int index = (t + 1) * stride * 2 - 1;
		if (index + stride < 2 * BLOCK_SIZE)
			scan_array[index + stride] += scan_array[index];
		__syncthreads();
	}

	// offset of 1 added to shift all elements right
	if (start + t + 1 < len)
		output[start + t + 1] = scan_array[t];
	if (start + BLOCK_SIZE + t + 1 < len)
		output[start + BLOCK_SIZE + t + 1] = scan_array[BLOCK_SIZE + t];

	if(t == 0)
	{
		output[0] = 0;
	}
}

__global__ void prescan_integer(int * input, int * output, int len) {
	// Load a segment of the input vector into shared memory
	__shared__ int scan_array[BLOCK_SIZE << 1];
	unsigned int t = threadIdx.x, start = 2 * blockIdx.x * BLOCK_SIZE;
	if (start + t < len)
		scan_array[t] = input[start + t];
	else
		scan_array[t] = 0;
	if (start + BLOCK_SIZE + t < len)
		scan_array[BLOCK_SIZE + t] = input[start + BLOCK_SIZE + t];
	else
		scan_array[BLOCK_SIZE + t] = 0;
	__syncthreads();

	// Reduction
	int stride;
	for (stride = 1; stride <= BLOCK_SIZE; stride <<= 1) {
		int index = (t + 1) * stride * 2 - 1;
		if (index < 2 * BLOCK_SIZE)
			scan_array[index] += scan_array[index - stride];
		__syncthreads();
	}

	// Post reduction
	for (stride = BLOCK_SIZE >> 1; stride; stride >>= 1) {
		int index = (t + 1) * stride * 2 - 1;
		if (index + stride < 2 * BLOCK_SIZE)
			scan_array[index + stride] += scan_array[index];
		__syncthreads();
	}

	if (start + t + 1 < len)
		output[start + t + 1] = scan_array[t];
	if (start + BLOCK_SIZE + t + 1 < len)
		output[start + BLOCK_SIZE + t + 1] = scan_array[BLOCK_SIZE + t];

	if (t == 0)
	{
		output[0] = 0;
	}
}*/

// From: http://stackoverflow.com/questions/30832033/is-prefix-scan-cuda-sample-code-in-gpugems3-correct
/*template <typename T>
__global__ void prescan(T *g_idata, T *g_odata, int n)
{
	extern __shared__ T temp[];  // allocated on invocation
	int thid = threadIdx.x;
	int offset = 1;
	temp[2 * thid] = g_idata[2 * thid]; // load input into shared memory
	temp[2 * thid + 1] = g_idata[2 * thid + 1];
	for (int d = n >> 1; d > 0; d >>= 1)                    // build sum in place up the tree
	{
		__syncthreads();
		if (thid < d)
		{
			int ai = offset*(2 * thid + 1) - 1;
			int bi = offset*(2 * thid + 2) - 1;
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}
	if (thid == 0) { temp[n - 1] = 0; } // clear the last element
	for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
	{
		offset >>= 1;
		__syncthreads();
		if (thid < d)
		{
			int ai = offset*(2 * thid + 1) - 1;
			int bi = offset*(2 * thid + 2) - 1;
			T t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();
	g_odata[2 * thid] = temp[2 * thid]; // write results to device memory
	g_odata[2 * thid + 1] = temp[2 * thid + 1];
}*/