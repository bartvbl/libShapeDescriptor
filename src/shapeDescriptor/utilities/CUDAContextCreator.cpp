#include <shapeDescriptor/shapeDescriptor.h>

#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
#include "helper_cuda.h"
#endif

#include <shapeDescriptor/libraryBuildSettings.h>
#include <iostream>

int ShapeDescriptor::createCUDAContext(int forceGPU)
{
#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
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

	std::cout << "CUDA context created on device " << chosenDeviceIndex << " (" << deviceWithMostMemory.name << ")" << std::endl;

	return chosenDeviceIndex;
#else
    throw std::runtime_error(ShapeDescriptor::cudaMissingErrorMessage);
#endif
}

void ShapeDescriptor::printGPUProperties(unsigned int deviceIndex) {
#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
	cudaDeviceProp deviceInfo;

	checkCudaErrors(cudaGetDeviceProperties(&deviceInfo, deviceIndex));

	std::cout << "This device supports CUDA Compute Capability v" << deviceInfo.major << "." << deviceInfo.minor << "." << std::endl;
	std::cout << std::endl;
	std::cout << "Other device info:" << std::endl;
	std::cout << "\t- Total global memory: " << deviceInfo.totalGlobalMem << std::endl;
	std::cout << "\t- Clock rate (KHz): " << deviceInfo.clockRate << std::endl;
	std::cout << "\t- Number of concurrent kernels: " << deviceInfo.concurrentKernels << std::endl;
	std::cout << "\t- Max grid size: (" << deviceInfo.maxGridSize[0] << ", " << deviceInfo.maxGridSize[1] << ", " << deviceInfo.maxGridSize[2] << ")" << std::endl;
	std::cout << "\t- Max threads per block dimension: (" << deviceInfo.maxThreadsDim[0] << ", " << deviceInfo.maxThreadsDim[1] << ", " << deviceInfo.maxThreadsDim[2] << ")" << std::endl;
	std::cout << "\t- Max threads per block: " << deviceInfo.maxThreadsPerBlock << std::endl;
	std::cout << "\t- Max threads per multiprocessor: " << deviceInfo.maxThreadsPerMultiProcessor << std::endl;
	std::cout << "\t- Number of multiprocessors: " << deviceInfo.multiProcessorCount << std::endl;
	std::cout << "\t- Number of registers per block: " << deviceInfo.regsPerBlock << std::endl;
	std::cout << "\t- Number of registers per multiprocessor: " << deviceInfo.regsPerMultiprocessor << std::endl;
	std::cout << "\t- Total constant memory: " << deviceInfo.totalConstMem << std::endl;
	std::cout << "\t- Warp size measured in threads: " << deviceInfo.warpSize << std::endl;
	std::cout << "\t- Single to double precision performance ratio: " << deviceInfo.singleToDoublePrecisionPerfRatio << std::endl;
	std::cout << "\t- Shared memory per block: " << deviceInfo.sharedMemPerBlock << std::endl;
	std::cout << "\t- Shared memory per multiprocessor: " << deviceInfo.sharedMemPerMultiprocessor << std::endl;
	std::cout << "\t- L2 Cache size: " << deviceInfo.l2CacheSize << std::endl;
	std::cout << std::endl;
#else
    throw std::runtime_error(ShapeDescriptor::cudaMissingErrorMessage);
#endif
}
