#include "meshSampler.cuh"

#include <shapeDescriptor/common/types/SampleBounds.h>
#include <shapeDescriptor/gpu/types/CudaLaunchDimensions.h>

#include <shapeDescriptor/gpu/types/array.h>
#include <shapeDescriptor/gpu/types/float3.h>

#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
#include "nvidia/helper_math.h"
#include "nvidia/helper_cuda.h"
#include <cuda_runtime_api.h>
#include <curand.h>
#include <curand_kernel.h>
#endif

#define SAMPLE_COEFFICIENT_THREAD_COUNT 4096

#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
__device__ __inline__ ShapeDescriptor::SampleBounds calculateSampleBounds(const ShapeDescriptor::gpu::array<float> &areaArray, int triangleIndex, int sampleCount) {
    ShapeDescriptor::SampleBounds sampleBounds;
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

__device__ __inline__ void lookupTriangleVertices(ShapeDescriptor::gpu::Mesh mesh, int triangleIndex, float3 (&triangleVertices)[3]) {
    assert(triangleIndex >= 0);
    assert((3 * triangleIndex) + 2 < mesh.vertexCount);

    unsigned int triangleBaseIndex = 3 * triangleIndex;

    triangleVertices[0].x = mesh.vertices_x[triangleBaseIndex];
    triangleVertices[0].y = mesh.vertices_y[triangleBaseIndex];
    triangleVertices[0].z = mesh.vertices_z[triangleBaseIndex];

    triangleVertices[1].x = mesh.vertices_x[triangleBaseIndex + 1];
    triangleVertices[1].y = mesh.vertices_y[triangleBaseIndex + 1];
    triangleVertices[1].z = mesh.vertices_z[triangleBaseIndex + 1];

    triangleVertices[2].x = mesh.vertices_x[triangleBaseIndex + 2];
    triangleVertices[2].y = mesh.vertices_y[triangleBaseIndex + 2];
    triangleVertices[2].z = mesh.vertices_z[triangleBaseIndex + 2];
}

__device__ __inline__ void lookupTriangleNormals(ShapeDescriptor::gpu::Mesh mesh, int triangleIndex, float3 (&triangleNormals)[3]) {
    assert(triangleIndex >= 0);
    assert((3 * triangleIndex) + 2 < mesh.vertexCount);

    unsigned int triangleBaseIndex = 3 * triangleIndex;

    triangleNormals[0].x = mesh.normals_x[triangleBaseIndex];
    triangleNormals[0].y = mesh.normals_y[triangleBaseIndex];
    triangleNormals[0].z = mesh.normals_z[triangleBaseIndex];

    triangleNormals[1].x = mesh.normals_x[triangleBaseIndex + 1];
    triangleNormals[1].y = mesh.normals_y[triangleBaseIndex + 1];
    triangleNormals[1].z = mesh.normals_z[triangleBaseIndex + 1];

    triangleNormals[2].x = mesh.normals_x[triangleBaseIndex + 2];
    triangleNormals[2].y = mesh.normals_y[triangleBaseIndex + 2];
    triangleNormals[2].z = mesh.normals_z[triangleBaseIndex + 2];
}


// One thread = One triangle
__global__ void calculateAreas(ShapeDescriptor::gpu::array<float> areaArray, ShapeDescriptor::gpu::Mesh mesh)
{
    int triangleIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (triangleIndex >= areaArray.length)
    {
        return;
    }
    float3 vertices[3];
    lookupTriangleVertices(mesh, triangleIndex, vertices);
    float3 v1 = vertices[1] - vertices[0];
    float3 v2 = vertices[2] - vertices[0];
    float area = length(cross(v1, v2)) / 2.0;
    areaArray.content[triangleIndex] = area;
}

__global__ void calculateCumulativeAreas(ShapeDescriptor::gpu::array<float> areaArray, ShapeDescriptor::gpu::array<float> device_cumulativeAreaArray) {
    int triangleIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (triangleIndex >= areaArray.length)
    {
        return;
    }

    float totalArea = 0;

    for(int i = 0; i <= triangleIndex; i++) {
        // Super inaccurate. Don't try this at home.
        totalArea += areaArray.content[i];
    }

    device_cumulativeAreaArray.content[triangleIndex] = totalArea;
}

__global__ void generateRandomSampleCoefficients(ShapeDescriptor::gpu::array<float2> coefficients, curandState *randomState, int sampleCount, size_t randomSeed) {
    int rawThreadIndex = threadIdx.x+blockDim.x*blockIdx.x;

    assert(rawThreadIndex < SAMPLE_COEFFICIENT_THREAD_COUNT);

    if(randomSeed == 0) {
        randomSeed = clock64();
    }

    // The addition of the thread index is overkill, but whatever. Randomness!
    size_t skipFactor = rawThreadIndex + (gridDim.x * blockDim.x);

    curand_init(randomSeed, skipFactor, 0, &randomState[rawThreadIndex]);

    for(int i = rawThreadIndex; i < sampleCount; i += blockDim.x * gridDim.x) {
        float v1 = curand_uniform(&(randomState[rawThreadIndex]));
        float v2 = curand_uniform(&(randomState[rawThreadIndex]));

        coefficients.content[i].x = v1;
        coefficients.content[i].y = v2;
    }
}

// One thread = One triangle
__global__ void sampleMesh(
        ShapeDescriptor::gpu::Mesh mesh,
        ShapeDescriptor::gpu::array<float> areaArray,
        ShapeDescriptor::gpu::PointCloud pointCloud,
        ShapeDescriptor::gpu::array<float2> coefficients,
        int sampleCount) {
    int triangleIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if(triangleIndex >= mesh.vertexCount / 3)
    {
        return;
    }

    float3 triangleVertices[3];
    lookupTriangleVertices(mesh, triangleIndex, triangleVertices);

    float3 triangleNormals[3];
    lookupTriangleNormals(mesh, triangleIndex, triangleNormals);

    ShapeDescriptor::SampleBounds bounds = calculateSampleBounds(areaArray, triangleIndex, sampleCount);

    for(int sample = 0; sample < bounds.sampleCount; sample++) {
        size_t sampleIndex = bounds.sampleStartIndex + sample;

        if(sampleIndex >= sampleCount) {
            continue;
        }

        float v1 = coefficients.content[sampleIndex].x;
        float v2 = coefficients.content[sampleIndex].y;

        float3 samplePoint =
                (1 - sqrt(v1)) * triangleVertices[0] +
                (sqrt(v1) * (1 - v2)) * triangleVertices[1] +
                (sqrt(v1) * v2) * triangleVertices[2];

        float3 sampleNormal =
                (1 - sqrt(v1)) * triangleNormals[0] +
                (sqrt(v1) * (1 - v2)) * triangleNormals[1] +
                (sqrt(v1) * v2) * triangleNormals[2];
        sampleNormal = normalize(sampleNormal);

        assert(sampleIndex < sampleCount);
        pointCloud.vertices.set(sampleIndex, samplePoint);
        pointCloud.normals.set(sampleIndex, sampleNormal);
    }
}

#endif

ShapeDescriptor::gpu::PointCloud ShapeDescriptor::utilities::sampleMesh(gpu::Mesh device_mesh, size_t sampleCount, size_t randomSamplingSeed, ShapeDescriptor::internal::MeshSamplingBuffers* internalSampleBuffers) {
#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
    size_t vertexCount = device_mesh.vertexCount;
    size_t triangleCount = vertexCount / 3;

    size_t areaArrayLength = triangleCount;
    size_t areaArraySize = areaArrayLength * sizeof(float);
    curandState* device_randomState;
    ShapeDescriptor::gpu::array<float2> device_coefficients;

    ShapeDescriptor::gpu::array<float> device_areaArray;
    ShapeDescriptor::gpu::array<float> device_cumulativeAreaArray;

    gpu::PointCloud device_pointCloud(sampleCount);

    checkCudaErrors(cudaMalloc(&device_areaArray.content, areaArraySize));
    checkCudaErrors(cudaMalloc(&device_cumulativeAreaArray.content, areaArraySize));
    checkCudaErrors(cudaMalloc(&device_randomState, sizeof(curandState) * (size_t)SAMPLE_COEFFICIENT_THREAD_COUNT));
    checkCudaErrors(cudaMalloc(&device_coefficients.content, sizeof(float2) * sampleCount));

    device_areaArray.length = (size_t) areaArrayLength;
    device_cumulativeAreaArray.length = (size_t) areaArrayLength;

    gpu::CudaLaunchDimensions areaSettings = calculateCudaLaunchDimensions(device_areaArray.length);
    gpu::CudaLaunchDimensions cumulativeAreaSettings = calculateCudaLaunchDimensions(device_areaArray.length);

    calculateAreas <<<areaSettings.blocksPerGrid, areaSettings.threadsPerBlock >>> (device_areaArray, device_mesh);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    calculateCumulativeAreas<<<cumulativeAreaSettings.blocksPerGrid, cumulativeAreaSettings.threadsPerBlock>>>(device_areaArray, device_cumulativeAreaArray);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    generateRandomSampleCoefficients<<<SAMPLE_COEFFICIENT_THREAD_COUNT / 32, 32>>>(device_coefficients, device_randomState, sampleCount, randomSamplingSeed);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    sampleMesh <<<areaSettings.blocksPerGrid, areaSettings.threadsPerBlock>>>(device_mesh, device_cumulativeAreaArray, device_pointCloud, device_coefficients, sampleCount);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    if(internalSampleBuffers != nullptr) {
        internalSampleBuffers->cumulativeAreaArray = device_cumulativeAreaArray;
    } else {
        cudaFree(device_cumulativeAreaArray.content);
    }

    cudaFree(device_areaArray.content);
    cudaFree(device_randomState);
    cudaFree(device_coefficients.content);

    return device_pointCloud;
#else
    throw std::runtime_error(ShapeDescriptor::cudaMissingErrorMessage);
#endif
}