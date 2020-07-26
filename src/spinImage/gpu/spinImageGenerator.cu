#include "spinImageGenerator.cuh"

#include "nvidia/helper_math.h"
#include "nvidia/helper_cuda.h"

#include "device_launch_parameters.h"
#include "cuda_runtime.h"

#include <cassert>
#include <iostream>
#include <chrono>
#include <map>

#include <spinImage/gpu/types/Mesh.h>
#include <spinImage/gpu/types/CudaLaunchDimensions.h>
#include <spinImage/utilities/setValue.cuh>
#include <spinImage/utilities/meshSampler.cuh>
#include <spinImage/utilities/dumpers/spinImageDumper.h>
#include <spinImage/gpu/types/PointCloud.h>
#include <spinImage/gpu/types/DeviceVertexList.cuh>
#include <spinImage/gpu/types/SampleBounds.h>

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

// @TODO: Descriptors are created on a vertex by vertex basis, as of yet not on an arbitrary point by point basis. 
// Feature points with high curvature tend to lie on edges. In triangle meshes, you need vertices to lie on these edges or corners to create the shape
// As such vertex by vertex might not be an entirely bad way of getting hold of corresponding features
// In addition, the argument can be made that since we're looking at one mesh only, the resolution is not expected to vary significantly
// between different features in the model.
// @TODO: Ensure a descriptor is calculated over multiple cubes if it exits the bounds of the current one
// @TODO: If necessary, add support for the support angle. Might not be needed here due to the relatively small spin image size.
// @TODO: Determine whether all coordinates checked agains the cube grid are in cube grid space.

// Run once for every vertex index
__global__ void createDescriptors(
        SpinImage::gpu::Mesh mesh,
        SpinImage::gpu::DeviceOrientedPoint* device_spinImageOrigins,
        SpinImage::gpu::PointCloud pointCloud,
        SpinImage::array<spinImagePixelType> descriptors,
        SpinImage::array<float> areaArray,
        size_t sampleCount,
        float oneOverSpinImagePixelWidth,
        float supportAngleCosine)
{
#define spinImageIndex blockIdx.x

	const SpinImage::gpu::DeviceOrientedPoint spinOrigin = device_spinImageOrigins[spinImageIndex];

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
                atomicAdd(&localSpinImage[valueIndex], (interPixelX) * (interPixelY));
            }

            if (baseSpinImageCoordinateX + 1 >= 0 &&
                baseSpinImageCoordinateX + 1 < spinImageWidthPixels &&
                baseSpinImageCoordinateY + 0 >= -halfSpinImageSizePixels &&
                baseSpinImageCoordinateY + 0 < halfSpinImageSizePixels)
            {
                size_t valueIndex = size_t(
                		(baseSpinImageCoordinateY + 0 + spinImageWidthPixels / 2) * spinImageWidthPixels +
						 baseSpinImageCoordinateX + 1);
                atomicAdd(&localSpinImage[valueIndex], (1.0f - interPixelX) * (interPixelY));
            }

            if (baseSpinImageCoordinateX + 1 >= 0 &&
                baseSpinImageCoordinateX + 1 < spinImageWidthPixels &&
                baseSpinImageCoordinateY + 1 >= -halfSpinImageSizePixels &&
                baseSpinImageCoordinateY + 1 < halfSpinImageSizePixels)
            {
                size_t valueIndex = size_t(
                		(baseSpinImageCoordinateY + 1 + spinImageWidthPixels / 2) * spinImageWidthPixels +
						 baseSpinImageCoordinateX + 1);
                atomicAdd(&localSpinImage[valueIndex], (1.0f - interPixelX) * (1.0f - interPixelY));
            }

            if (baseSpinImageCoordinateX + 0 >= 0 &&
                baseSpinImageCoordinateX + 0 < spinImageWidthPixels &&
                baseSpinImageCoordinateY + 1 >= -halfSpinImageSizePixels &&
                baseSpinImageCoordinateY + 1 < halfSpinImageSizePixels)
            {
                size_t valueIndex = size_t(
                		(baseSpinImageCoordinateY + 1 + spinImageWidthPixels / 2) * spinImageWidthPixels +
						 baseSpinImageCoordinateX + 0);
                atomicAdd(&localSpinImage[valueIndex], (interPixelX) * (1.0f - interPixelY));
            }
		}
	}

	__syncthreads();

	// Copy final image into memory

	size_t imageBaseIndex = size_t(spinImageIndex) * spinImageWidthPixels * spinImageWidthPixels;
    for(size_t i = threadIdx.x; i < spinImageWidthPixels * spinImageWidthPixels; i += blockDim.x) {
        descriptors.content[imageBaseIndex + i] = localSpinImage[i];
    }

}

SpinImage::array<spinImagePixelType> SpinImage::gpu::generateSpinImages(
        Mesh device_mesh,
        array<DeviceOrientedPoint> device_spinImageOrigins,
        float spinImageWidth,
        size_t sampleCount,
        float supportAngleDegrees,
        size_t randomSamplingSeed,
        SpinImage::debug::SIExecutionTimes* executionTimes)
{
    auto totalExecutionTimeStart = std::chrono::steady_clock::now();

    size_t imageCount = device_spinImageOrigins.length;

    size_t descriptorBufferLength = imageCount * spinImageWidthPixels * spinImageWidthPixels;
	size_t descriptorBufferSize = sizeof(float) * descriptorBufferLength;

	array<spinImagePixelType> device_descriptors;

	float supportAngleCosine = float(std::cos(supportAngleDegrees * (M_PI / 180.0)));

	// -- Initialisation --
	auto initialisationStart = std::chrono::steady_clock::now();

		checkCudaErrors(cudaMalloc(&device_descriptors.content, descriptorBufferSize));

		device_descriptors.length = imageCount;

		CudaLaunchDimensions valueSetSettings = calculateCudaLaunchDimensions(descriptorBufferLength);

		setValue <spinImagePixelType><<<valueSetSettings.blocksPerGrid, valueSetSettings.threadsPerBlock >>> (device_descriptors.content, descriptorBufferLength, 0);
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
	            float(spinImageWidthPixels)/spinImageWidth,
	            supportAngleCosine);
	    checkCudaErrors(cudaDeviceSynchronize());
	    checkCudaErrors(cudaGetLastError());

	std::chrono::milliseconds generationDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - generationStart);

	// -- Cleanup --

	checkCudaErrors(cudaFree(device_cumulativeAreaArray.content));
	device_pointCloud.vertices.free();
	device_pointCloud.normals.free();

    std::chrono::milliseconds totalExecutionDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - totalExecutionTimeStart);

    if(executionTimes != nullptr) {
        executionTimes->totalExecutionTimeSeconds = double(totalExecutionDuration.count()) / 1000.0;
        executionTimes->initialisationTimeSeconds = double(initialisationDuration.count()) / 1000.0;
        executionTimes->meshSamplingTimeSeconds = double(meshSamplingDuration.count()) / 1000.0;
        executionTimes->generationTimeSeconds = double(generationDuration.count()) / 1000.0;
	}

	return device_descriptors;
}

