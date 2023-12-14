#include <shapeDescriptor/shapeDescriptor.h>
#include <stdexcept>


#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
#include <helper_cuda.h>
#include <cuda_runtime.h>

// GPUs have optimised memory transaction channels for aligned writes with a power of 2 in size
struct chunk4 { unsigned int content = 0; };
struct chunk8 { unsigned long long content = 0; };
struct chunk16 { unsigned long long content[2] = {0, 0}; };

template<typename valueType>
__global__ void setValueKernel(valueType* target, size_t length, valueType value)
{
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < length)
    {
        target[index] = value;
    }
}

__global__ void setByteArrayValue(char* target, size_t length, char* value, unsigned int valueSize)
{
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < length)
    {
        for(unsigned int i = 0; i < valueSize; i++) {
            target[index * valueSize + i] = value[i];
        }
    }
}
#endif

void ShapeDescriptor::internal::gpuMemsetMultibyte(char *array, size_t length, const char *value, size_t valueSize) {
#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
    unsigned int batchSize = 32;
    size_t count = (length / batchSize) + 1;

    chunk4 copiedValue4;
    chunk8 copiedValue8;
    chunk16 copiedValue16;

    switch(valueSize) {
        case 4:
            copiedValue4 = {*reinterpret_cast<const unsigned int*>(value)};
            setValueKernel<chunk4><<<count, batchSize>>>(reinterpret_cast<chunk4*>(array), length, copiedValue4);
            checkCudaErrors(cudaDeviceSynchronize());
            break;
        case 8:
            copiedValue8 = {*reinterpret_cast<const unsigned long long*>(value)};
            setValueKernel<chunk8><<<count, batchSize>>>(reinterpret_cast<chunk8*>(array), length, copiedValue8);
            checkCudaErrors(cudaDeviceSynchronize());
            break;
        case 16:
            copiedValue16 = {*reinterpret_cast<const unsigned long long*>(value),
                             *(reinterpret_cast<const unsigned long long*>(value) + 1)};
            setValueKernel<chunk16><<<count, batchSize>>>(reinterpret_cast<chunk16*>(array), length, copiedValue16);
            checkCudaErrors(cudaDeviceSynchronize());
            break;
        default:
            char* gpuValue;
            checkCudaErrors(cudaMalloc(&gpuValue, valueSize));
            checkCudaErrors(cudaMemcpy(gpuValue, value, valueSize, cudaMemcpyHostToDevice));
            setByteArrayValue<<<count, batchSize>>>(array, length, gpuValue, valueSize);
            checkCudaErrors(cudaDeviceSynchronize());
            checkCudaErrors(cudaFree(gpuValue));
    }
#else
    throw std::runtime_error(ShapeDescriptor::cudaMissingErrorMessage);
#endif
}