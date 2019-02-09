#include "CUDAContextCreator.h"

cudaDeviceProp createCUDAContext(int forceGPU = -1)
{
	int deviceCount;
	checkCudaErrors(cudaGetDeviceCount(&deviceCount));

	size_t maxAvailableMemory = 0;
	cudaDeviceProp deviceWithMostMemory;
	int chosenDeviceIndex = 0;
	
	for(int i = 0; i < deviceCount; i++)
	{
		cudaDeviceProp deviceProperties;
		checkCudaErrors(cudaGetDeviceProperties(&deviceProperties, i));

		if(deviceProperties.totalGlobalMem > maxAvailableMemory)
		{
			maxAvailableMemory = deviceProperties.totalGlobalMem;
			deviceWithMostMemory = deviceProperties;
			chosenDeviceIndex = i;
		}
	}

	if(forceGPU != -1) {
		chosenDeviceIndex = forceGPU;
	}

	checkCudaErrors(cudaGetDeviceProperties(&deviceWithMostMemory, chosenDeviceIndex));

	checkCudaErrors(cudaSetDevice(chosenDeviceIndex));

	return deviceWithMostMemory;
}

void printGPUProperties() {
	std::cout << "This device supports CUDA Compute Capability v" << deviceWithMostMemory.major << "." << deviceWithMostMemory.minor << "." << std::endl;
	std::cout << std::endl;
	std::cout << "Other device info:" << std::endl;
	std::cout << "\t- Total global memory: " << deviceWithMostMemory.totalGlobalMem << std::endl;
	std::cout << "\t- Clock rate (KHz): " << deviceWithMostMemory.clockRate << std::endl;
	std::cout << "\t- Number of concurrent kernels: " << deviceWithMostMemory.concurrentKernels << std::endl;
	std::cout << "\t- Max grid size: (" << deviceWithMostMemory.maxGridSize[0] << ", " << deviceWithMostMemory.maxGridSize[1] << ", " << deviceWithMostMemory.maxGridSize[2] << ")" << std::endl;
	std::cout << "\t- Max threads per block dimension: (" << deviceWithMostMemory.maxThreadsDim[0] << ", " << deviceWithMostMemory.maxThreadsDim[1] << ", " << deviceWithMostMemory.maxThreadsDim[2] << ")" << std::endl;
	std::cout << "\t- Max threads per block: " << deviceWithMostMemory.maxThreadsPerBlock << std::endl;
	std::cout << "\t- Max threads per multiprocessor: " << deviceWithMostMemory.maxThreadsPerMultiProcessor << std::endl;
	std::cout << "\t- Number of multiprocessors: " << deviceWithMostMemory.multiProcessorCount << std::endl;
	std::cout << "\t- Number of registers per block: " << deviceWithMostMemory.regsPerBlock << std::endl;
	std::cout << "\t- Number of registers per multiprocessor: " << deviceWithMostMemory.regsPerMultiprocessor << std::endl;
	std::cout << "\t- Total constant memory: " << deviceWithMostMemory.totalConstMem << std::endl;
	std::cout << "\t- Warp size measured in threads: " << deviceWithMostMemory.warpSize << std::endl;
	std::cout << "\t- Single to double precision performance ratio: " << deviceWithMostMemory.singleToDoublePrecisionPerfRatio << std::endl;
	std::cout << "\t- Shared memory per block: " << deviceWithMostMemory.sharedMemPerBlock << std::endl;
	std::cout << "\t- Shared memory per multiprocessor: " << deviceWithMostMemory.sharedMemPerMultiprocessor << std::endl;
	std::cout << "\t- L2 Cache size: " << deviceWithMostMemory.l2CacheSize << std::endl;
	std::cout << std::endl;
}