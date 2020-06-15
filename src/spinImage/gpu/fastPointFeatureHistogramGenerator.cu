#include "fastPointFeatureHistogramGenerator.cuh"
#include <iostream>
#include <pcl/gpu/features/features.hpp>
#include <nvidia/helper_cuda.h>
#include <cuda_runtime.h>
#include <spinImage/gpu/types/PointCloud.h>
#include <spinImage/utilities/meshSampler.cuh>
#include <chrono>

__global__ void reformatVertexBuffer(SpinImage::gpu::DeviceVertexList inputList, pcl::gpu::Feature::PointType* output, size_t count) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index >= count) {
        return;
    }

    float3 vertex = inputList.at(index);
    output[index].x = vertex.x;
    output[index].y = vertex.y;
    output[index].z = vertex.z;
};

__global__ void reformatOrigins(SpinImage::gpu::DeviceOrientedPoint* origins, pcl::gpu::Feature::PointType* vertices, pcl::gpu::Feature::PointType* normals, size_t count) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index >= count) {
        return;
    }

    float3 vertex = origins[index].vertex;
    float3 normal = origins[index].normal;

    vertices[index].x = vertex.x;
    vertices[index].y = vertex.y;
    vertices[index].z = vertex.z;

    normals[index].x = normal.x;
    normals[index].y = normal.y;
    normals[index].z = normal.z;
}



SpinImage::gpu::FPFHHistograms SpinImage::gpu::generateFPFHHistograms(
        Mesh device_mesh,
        array<SpinImage::gpu::DeviceOrientedPoint> device_origins,
        float supportRadius,
        unsigned int maxNeighbours,
        size_t sampleCount,
        size_t randomSamplingSeed,
        SpinImage::debug::FPFHRunInfo* runInfo)
{
    auto totalExecutionTimeStart = std::chrono::steady_clock::now();

    gpu::PointCloud pointCloud = SpinImage::utilities::sampleMesh(device_mesh, sampleCount, randomSamplingSeed);

    // PCL uses an inefficient memory layout for its GPU input buffers: float3[]
    // Threads can only do 4, 8, or 16 byte requests for fast streaming,
    // so this means you need to do 3 separate requests,
    // each making only use of 33% of the available memory bandwidth.
    // Still, the most fair comparison is using PCL as-is if we're saying in the paper that we're using their implementation
    // That requires as a first pre-processing step to reformat the input vertices and normals
    pcl::gpu::Feature::PointType* device_reformatted_vertices;
    pcl::gpu::Feature::PointType* device_reformatted_origins_vertices;
    pcl::gpu::Feature::PointType* device_reformatted_origins_normals;

    checkCudaErrors(cudaMalloc(&device_reformatted_vertices, sampleCount * sizeof(pcl::gpu::Feature::PointType)));
    checkCudaErrors(cudaMalloc(&device_reformatted_origins_vertices, device_origins.length * sizeof(pcl::gpu::Feature::PointType)));
    checkCudaErrors(cudaMalloc(&device_reformatted_origins_normals, device_origins.length * sizeof(pcl::gpu::Feature::PointType)));

    // Copy data
    reformatOrigins<<<(unsigned int) ((device_origins.length / 32) + 1), 32>>>(
            device_origins.content,
            device_reformatted_origins_vertices,
            device_reformatted_origins_normals,
            device_origins.length);
    reformatVertexBuffer<<<(unsigned int) ((sampleCount / 32) + 1), 32>>>(
            pointCloud.vertices,
            device_reformatted_vertices,
            sampleCount);

    checkCudaErrors(cudaDeviceSynchronize());


    //uploading data to GPU
    pcl::gpu::FPFHEstimation::PointCloud cloud_gpu(device_reformatted_origins_vertices, device_origins.length);
    pcl::gpu::FPFHEstimation::Normals normals_gpu(device_reformatted_origins_normals, device_origins.length);
    pcl::gpu::FPFHEstimation::PointCloud surface_gpu(device_reformatted_vertices, device_mesh.vertexCount);



    //GPU call
    pcl::gpu::FPFHEstimation fe_gpu;
    fe_gpu.setInputCloud (cloud_gpu);
    fe_gpu.setInputNormals (normals_gpu);
    fe_gpu.setSearchSurface(surface_gpu);
    fe_gpu.setRadiusSearch(supportRadius, int(maxNeighbours));

    pcl::gpu::DeviceArray2D<pcl::FPFHSignature33> device_fpfhSignatures;
    fe_gpu.compute(device_fpfhSignatures);

    checkCudaErrors(cudaFree(device_reformatted_vertices));
    checkCudaErrors(cudaFree(device_reformatted_origins_vertices));
    checkCudaErrors(cudaFree(device_reformatted_origins_normals));

    pointCloud.free();

    // Making a persistent copy of the generated descriptors
    SpinImage::gpu::FPFHHistograms device_histograms;

    size_t outputHistogramsSize = device_origins.length * sizeof(pcl::FPFHSignature33);

    checkCudaErrors(cudaMalloc(&device_histograms.histograms, outputHistogramsSize));
    checkCudaErrors(cudaMemcpy(device_histograms.histograms, device_fpfhSignatures.ptr(), outputHistogramsSize, cudaMemcpyDeviceToDevice));

    std::chrono::milliseconds totalExecutionDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - totalExecutionTimeStart);

    if(runInfo != nullptr) {
        runInfo->totalExecutionTimeSeconds = double(totalExecutionDuration.count()) / 1000.0;
    }

    return device_histograms;
}