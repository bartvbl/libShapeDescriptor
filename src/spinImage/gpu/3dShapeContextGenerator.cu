#include <spinImage/gpu/types/SampleBounds.h>
#include <spinImage/gpu/types/PointCloud.h>
#include <spinImage/gpu/types/CudaLaunchDimensions.h>
#include <spinImage/utilities/meshSampler.cuh>
#include <spinImage/utilities/setValue.cuh>
#include <chrono>
#include <cuda_runtime.h>
#include <nvidia/helper_cuda.h>
#include "3dShapeContextGenerator.cuh"


__device__ __inline__ SpinImage::SampleBounds calculateSampleBounds(const SpinImage::array<float> &areaArray, int triangleIndex, int sampleCount) {
    SpinImage::SampleBounds sampleBounds;
    float maxArea = areaArray.content[areaArray.length - 1];
    float areaStepSize = maxArea / (float)sampleCount;

    if (triangleIndex == 0)
    {
        sampleBounds.areaStart = 0;
        sampleBounds.areaEnd = areaArray.content[0];
    }
    else
    {
        sampleBounds.areaStart = areaArray.content[triangleIndex - 1];
        sampleBounds.areaEnd = areaArray.content[triangleIndex];
    }

    size_t firstIndexInRange = (size_t) (sampleBounds.areaStart / areaStepSize) + 1;
    size_t lastIndexInRange = (size_t) (sampleBounds.areaEnd / areaStepSize);

    sampleBounds.sampleCount = lastIndexInRange - firstIndexInRange + 1; // Offset is needed to ensure bounds are correct
    sampleBounds.sampleStartIndex = firstIndexInRange - 1;

    return sampleBounds;
}

// Run once for every vertex index
__global__ void createDescriptors(
        SpinImage::gpu::Mesh mesh,
        SpinImage::gpu::DeviceOrientedPoint* device_spinImageOrigins,
        SpinImage::gpu::PointCloud pointCloud,
        SpinImage::array<shapeContextBinType> descriptors,
        SpinImage::array<float> areaArray,
        size_t sampleCount,
        float oneOverSpinImagePixelWidth)
{
#define descriptorIndex blockIdx.x

    const SpinImage::gpu::DeviceOrientedPoint spinOrigin = device_spinImageOrigins[descriptorIndex];

    const float3 vertex = spinOrigin.vertex;
    const float3 normal = spinOrigin.normal;

    __shared__ float localSpinImage[spinImageWidthPixels * spinImageWidthPixels];
    for(int i = threadIdx.x; i < spinImageWidthPixels * spinImageWidthPixels; i += blockDim.x) {
        localSpinImage[i] = 0;
    }

    __syncthreads();

    for (int triangleIndex = threadIdx.x; triangleIndex < mesh.vertexCount / 3; triangleIndex += blockDim.x)
    {
        SpinImage::SampleBounds bounds = calculateSampleBounds(areaArray, triangleIndex, sampleCount);

        for(unsigned int sample = 0; sample < bounds.sampleCount; sample++)
        {
            size_t sampleIndex = bounds.sampleStartIndex + sample;

            if(sampleIndex >= sampleCount) {
                printf("Sample %i/%i/%i was skipped.\n", sampleIndex, bounds.sampleCount, sampleCount);
                continue;
            }

            float3 samplePoint = pointCloud.vertices.at(sampleIndex);
            float3 sampleNormal = pointCloud.normals.at(sampleIndex);

            // DESCRIPTOR GOES HERE
        }
    }

    __syncthreads();

    // Copy final image into memory

    size_t imageBaseIndex = size_t(descriptorIndex) * spinImageWidthPixels * spinImageWidthPixels;
    for(size_t i = threadIdx.x; i < spinImageWidthPixels * spinImageWidthPixels; i += blockDim.x) {
        descriptors.content[imageBaseIndex + i] = localSpinImage[i];
    }

}

SpinImage::array<shapeContextBinType> SpinImage::gpu::generate3DSCDescriptors(
        Mesh device_mesh,
        array<DeviceOrientedPoint> device_spinImageOrigins,
        float supportRadius,
        size_t sampleCount,
        size_t randomSamplingSeed,
        SpinImage::debug::SCRunInfo* runInfo) {
    auto totalExecutionTimeStart = std::chrono::steady_clock::now();

    size_t imageCount = device_spinImageOrigins.length;

    size_t descriptorBufferLength = imageCount * spinImageWidthPixels * spinImageWidthPixels;
    size_t descriptorBufferSize = sizeof(shapeContextBinType) * descriptorBufferLength;

    array<shapeContextBinType> device_descriptors;

    // -- Initialisation --
    auto initialisationStart = std::chrono::steady_clock::now();

    checkCudaErrors(cudaMalloc(&device_descriptors.content, descriptorBufferSize));

    device_descriptors.length = imageCount;

    CudaLaunchDimensions valueSetSettings = calculateCudaLaunchDimensions(descriptorBufferLength);

    setValue <shapeContextBinType><<<valueSetSettings.blocksPerGrid, valueSetSettings.threadsPerBlock >>> (device_descriptors.content, descriptorBufferLength, 0);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    std::chrono::milliseconds initialisationDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - initialisationStart);

    // -- Mesh Sampling --
    auto meshSamplingStart = std::chrono::steady_clock::now();

    SpinImage::internal::MeshSamplingBuffers sampleBuffers;
    PointCloud device_pointCloud = SpinImage::utilities::sampleMesh(device_mesh, sampleCount, randomSamplingSeed, &sampleBuffers);
    array<float> device_cumulativeAreaArray = sampleBuffers.cumulativeAreaArray;

    std::chrono::milliseconds meshSamplingDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - meshSamplingStart);

    // -- Spin Image Generation --
    auto generationStart = std::chrono::steady_clock::now();

    createDescriptors <<<imageCount, 416>>>(
            device_mesh,
                    device_spinImageOrigins.content,
                    device_pointCloud,
                    device_descriptors,
                    device_cumulativeAreaArray,
                    sampleCount,
                    float(spinImageWidthPixels)/supportRadius);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    std::chrono::milliseconds generationDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - generationStart);

    // -- Cleanup --

    checkCudaErrors(cudaFree(device_cumulativeAreaArray.content));
    device_pointCloud.vertices.free();
    device_pointCloud.normals.free();

    std::chrono::milliseconds totalExecutionDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - totalExecutionTimeStart);

    if(runInfo != nullptr) {
        runInfo->totalExecutionTimeSeconds = double(totalExecutionDuration.count()) / 1000.0;
        runInfo->initialisationTimeSeconds = double(initialisationDuration.count()) / 1000.0;
        runInfo->meshSamplingTimeSeconds = double(meshSamplingDuration.count()) / 1000.0;
        runInfo->generationTimeSeconds = double(generationDuration.count()) / 1000.0;
    }

    return device_descriptors;
}