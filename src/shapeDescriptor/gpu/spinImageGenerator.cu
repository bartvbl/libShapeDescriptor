#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
#include "helper_math.h"
#include "helper_cuda.h"
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#endif

#include <cassert>
#include <iostream>
#include <chrono>
#include <map>

#include <shapeDescriptor/shapeDescriptor.h>

#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
__device__ __inline__ float2 calculateAlphaBeta(float3 spinVertex, float3 spinNormal, float3 point)
{
	// Using the projective properties of the dot product, an arbitrary point
	// can be projected on to the line defined by the vertex around which the spin image is generated
	// along with its surface normal.
	// The formula I used here yields a factor representing the number of times the normal vector should
	// be added to the spin vertex to get the closest point. However, since we are only interested in
	// the distance, we can operate on the distance value directly. 
	float beta = dot(point - spinVertex, spinNormal) / dot(spinNormal, spinNormal);


	float3 projectedPoint = spinVertex + beta * spinNormal;
	float3 delta = projectedPoint - point;
	float alpha = length(delta);

	float2 alphabeta = make_float2(alpha, beta);

	return alphabeta;
}

// Run once for every vertex index
__global__ void createDescriptors(
        ShapeDescriptor::OrientedPoint* device_spinImageOrigins,
        ShapeDescriptor::gpu::PointCloud pointCloud,
        ShapeDescriptor::gpu::array<ShapeDescriptor::SpinImageDescriptor> descriptors,
        float oneOverSpinImagePixelWidth,
        float supportAngleCosine)
{
#define spinImageIndex blockIdx.x

	const ShapeDescriptor::OrientedPoint spinOrigin = device_spinImageOrigins[spinImageIndex];

	const float3 vertex = spinOrigin.vertex;
	const float3 normal = spinOrigin.normal;

	__shared__ ShapeDescriptor::SpinImageDescriptor localSpinImage;
	for(int i = threadIdx.x; i < spinImageWidthPixels * spinImageWidthPixels; i += blockDim.x) {
	    localSpinImage.contents[i] = 0;
	}

	__syncthreads();

	for (int sampleIndex = threadIdx.x; sampleIndex < pointCloud.pointCount; sampleIndex += blockDim.x)
	{
        float3 samplePoint = pointCloud.vertices.at(sampleIndex);
        float3 sampleNormal = pointCloud.normals.at(sampleIndex);

        float sampleAngleCosine = dot(sampleNormal, normal);

        if(sampleAngleCosine < supportAngleCosine) {
            // Discard the sample
            continue;
        }

        float2 sampleAlphaBeta = calculateAlphaBeta(vertex, normal, samplePoint);

        float floatSpinImageCoordinateX = (sampleAlphaBeta.x * oneOverSpinImagePixelWidth);
        float floatSpinImageCoordinateY = (sampleAlphaBeta.y * oneOverSpinImagePixelWidth);

        int baseSpinImageCoordinateX = (int) floorf(floatSpinImageCoordinateX);
        int baseSpinImageCoordinateY = (int) floorf(floatSpinImageCoordinateY);

        float interPixelX = floatSpinImageCoordinateX - floorf(floatSpinImageCoordinateX);
        float interPixelY = floatSpinImageCoordinateY - floorf(floatSpinImageCoordinateY);

        const int halfSpinImageSizePixels = spinImageWidthPixels / 2;

        if (baseSpinImageCoordinateX + 0 >= 0 &&
            baseSpinImageCoordinateX + 0 < spinImageWidthPixels &&
            baseSpinImageCoordinateY + 0 >= -halfSpinImageSizePixels &&
            baseSpinImageCoordinateY + 0 < halfSpinImageSizePixels)
        {
            size_t valueIndex = size_t(
                    (baseSpinImageCoordinateY + 0 + spinImageWidthPixels / 2) * spinImageWidthPixels +
                     baseSpinImageCoordinateX + 0);
            atomicAdd(&localSpinImage.contents[valueIndex], (interPixelX) * (interPixelY));
        }

        if (baseSpinImageCoordinateX + 1 >= 0 &&
            baseSpinImageCoordinateX + 1 < spinImageWidthPixels &&
            baseSpinImageCoordinateY + 0 >= -halfSpinImageSizePixels &&
            baseSpinImageCoordinateY + 0 < halfSpinImageSizePixels)
        {
            size_t valueIndex = size_t(
                    (baseSpinImageCoordinateY + 0 + spinImageWidthPixels / 2) * spinImageWidthPixels +
                     baseSpinImageCoordinateX + 1);
            atomicAdd(&localSpinImage.contents[valueIndex], (1.0f - interPixelX) * (interPixelY));
        }

        if (baseSpinImageCoordinateX + 1 >= 0 &&
            baseSpinImageCoordinateX + 1 < spinImageWidthPixels &&
            baseSpinImageCoordinateY + 1 >= -halfSpinImageSizePixels &&
            baseSpinImageCoordinateY + 1 < halfSpinImageSizePixels)
        {
            size_t valueIndex = size_t(
                    (baseSpinImageCoordinateY + 1 + spinImageWidthPixels / 2) * spinImageWidthPixels +
                     baseSpinImageCoordinateX + 1);
            atomicAdd(&localSpinImage.contents[valueIndex], (1.0f - interPixelX) * (1.0f - interPixelY));
        }

        if (baseSpinImageCoordinateX + 0 >= 0 &&
            baseSpinImageCoordinateX + 0 < spinImageWidthPixels &&
            baseSpinImageCoordinateY + 1 >= -halfSpinImageSizePixels &&
            baseSpinImageCoordinateY + 1 < halfSpinImageSizePixels)
        {
            size_t valueIndex = size_t(
                    (baseSpinImageCoordinateY + 1 + spinImageWidthPixels / 2) * spinImageWidthPixels +
                     baseSpinImageCoordinateX + 0);
            atomicAdd(&localSpinImage.contents[valueIndex], (interPixelX) * (1.0f - interPixelY));
        }
	}

	__syncthreads();

	// Copy final image into memory
    for(size_t i = threadIdx.x; i < spinImageWidthPixels * spinImageWidthPixels; i += blockDim.x) {
        descriptors.content[spinImageIndex].contents[i] = localSpinImage.contents[i];
    }
}
#endif

ShapeDescriptor::gpu::array<ShapeDescriptor::SpinImageDescriptor> ShapeDescriptor::generateSpinImages(
        ShapeDescriptor::gpu::PointCloud device_pointCloud,
        ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> device_descriptorOrigins,
        float supportRadius,
        float supportAngleDegrees,
        ShapeDescriptor::SIExecutionTimes* executionTimes) {
#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
    auto totalExecutionTimeStart = std::chrono::steady_clock::now();

    size_t imageCount = device_descriptorOrigins.length;

	size_t descriptorBufferSize = imageCount * sizeof(ShapeDescriptor::SpinImageDescriptor);

	ShapeDescriptor::gpu::array<ShapeDescriptor::SpinImageDescriptor> device_descriptors;

	float supportAngleCosine = float(std::cos(supportAngleDegrees * (M_PI / 180.0)));

	// -- Initialisation --
	auto initialisationStart = std::chrono::steady_clock::now();

		checkCudaErrors(cudaMalloc(&device_descriptors.content, descriptorBufferSize));

		device_descriptors.length = imageCount;

		cudaMemset(device_descriptors.content, 0, descriptorBufferSize);

	std::chrono::milliseconds initialisationDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - initialisationStart);


	// -- Spin Image Generation --
	auto generationStart = std::chrono::steady_clock::now();

	    createDescriptors <<<imageCount, 416>>>(
	            device_descriptorOrigins.content,
                device_pointCloud,
	            device_descriptors,
	            float(spinImageWidthPixels)/supportRadius,
	            supportAngleCosine);
	    checkCudaErrors(cudaDeviceSynchronize());
	    checkCudaErrors(cudaGetLastError());

	std::chrono::milliseconds generationDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - generationStart);

    std::chrono::milliseconds totalExecutionDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - totalExecutionTimeStart);

    if(executionTimes != nullptr) {
        executionTimes->totalExecutionTimeSeconds = double(totalExecutionDuration.count()) / 1000.0;
        executionTimes->initialisationTimeSeconds = double(initialisationDuration.count()) / 1000.0;
        executionTimes->generationTimeSeconds = double(generationDuration.count()) / 1000.0;
	}

	return device_descriptors;
#else
    throw std::runtime_error(ShapeDescriptor::cudaMissingErrorMessage);
#endif
}

