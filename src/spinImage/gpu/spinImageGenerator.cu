#include "spinImageGenerator.cuh"

#include "nvidia/shfl_scan.cuh"
#include "nvidia/helper_math.h"
#include "nvidia/helper_cuda.h"

#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include <curand.h>
#include <curand_kernel.h>

#include <assert.h>
#include <iostream>
#include <chrono>
#include <map>

#include <spinImage/gpu/types/DeviceMesh.h>
#include <spinImage/gpu/types/CudaLaunchDimensions.h>
#include <spinImage/gpu/setValue.cuh>
#include <spinImage/utilities/dumpers/spinImageDumper.h>

#define SAMPLE_COEFFICIENT_THREAD_COUNT 4096



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

__device__ __inline__ void lookupTriangleVertices(DeviceMesh mesh, int triangleIndex, float3 (&triangleVertices)[3]) {
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

	assert(!isnan(triangleVertices[0].x) && !isnan(triangleVertices[0].y) && !isnan(triangleVertices[0].z));
	assert(!isnan(triangleVertices[1].x) && !isnan(triangleVertices[1].y) && !isnan(triangleVertices[1].z));
	assert(!isnan(triangleVertices[2].x) && !isnan(triangleVertices[2].y) && !isnan(triangleVertices[2].z));
}

struct SampleBounds {
	size_t sampleCount;
	float areaStart;
	float areaEnd;
	size_t sampleStartIndex;
};

__device__ __inline__ SampleBounds calculateSampleBounds(const array<float> &areaArray, int triangleIndex, int sampleCount) {
	SampleBounds sampleBounds;
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

// One thread = One triangle
__global__ void calculateAreas(floatArray areaArray, DeviceMesh mesh)
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

__global__ void calculateCumulativeAreas(floatArray areaArray, floatArray device_cumulativeAreaArray) {
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

__global__ void generateRandomSampleCoefficients(array<float2> coefficients, curandState *randomState, int sampleCount) {
	int rawThreadIndex = threadIdx.x+blockDim.x*blockIdx.x;

	assert(rawThreadIndex < SAMPLE_COEFFICIENT_THREAD_COUNT);

	curand_init(clock64(), rawThreadIndex, 0, &randomState[rawThreadIndex]);

	for(int i = rawThreadIndex; i < sampleCount; i += blockDim.x * gridDim.x) {
		float v1 = curand_uniform(&(randomState[rawThreadIndex]));
		float v2 = curand_uniform(&(randomState[rawThreadIndex]));

		coefficients.content[i].x = v1;
		coefficients.content[i].y = v2;
	}
}

// One thread = One triangle
__global__ void sampleMesh(DeviceMesh mesh, array<float> areaArray, array<float3> pointSamples,
						   array<float2> coefficients, int sampleCount) {
	int triangleIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if(triangleIndex >= mesh.vertexCount / 3)
	{
		return;
	}

	float3 triangleVertices[3];
	lookupTriangleVertices(mesh, triangleIndex, triangleVertices);

	SampleBounds bounds = calculateSampleBounds(areaArray, triangleIndex, sampleCount);

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

		assert(sampleIndex < sampleCount);
		assert(sampleIndex < pointSamples.length);
		pointSamples.content[sampleIndex] = samplePoint;
	}
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
__global__ void createDescriptors(DeviceMesh mesh, array<float3> pointSamples, array<spinImagePixelType> descriptors, array<float> areaArray, int sampleCount, float oneOverSpinImagePixelWidth)
{
#define spinImageIndexIndex blockIdx.x

	if(spinImageIndexIndex >= mesh.vertexCount)
	{
		return;
	}

	float3 vertex;
	float3 normal;

	vertex.x = mesh.vertices_x[spinImageIndexIndex];
	vertex.y = mesh.vertices_y[spinImageIndexIndex];
	vertex.z = mesh.vertices_z[spinImageIndexIndex];

	normal.x = mesh.normals_x[spinImageIndexIndex];
	normal.y = mesh.normals_y[spinImageIndexIndex];
	normal.z = mesh.normals_z[spinImageIndexIndex];

	__shared__ float localSpinImage[spinImageWidthPixels * spinImageWidthPixels];
	for(int i = threadIdx.x; i < spinImageWidthPixels * spinImageWidthPixels; i += blockDim.x) {
	    localSpinImage[i] = 0;
	}

	__syncthreads();

	for (int triangleIndex = threadIdx.x; triangleIndex < mesh.vertexCount / 3; triangleIndex += blockDim.x)
	{
		SampleBounds bounds = calculateSampleBounds(areaArray, triangleIndex, sampleCount);

		for(unsigned int sample = 0; sample < bounds.sampleCount; sample++)
		{
			size_t sampleIndex = bounds.sampleStartIndex + sample;

			if(sampleIndex >= sampleCount) {
				printf("Sample %i/%i/%i was skipped.\n", sampleIndex, bounds.sampleCount, sampleCount);
				continue;
			}

			assert(sampleIndex < pointSamples.length);

			float3 samplePoint = pointSamples.content[sampleIndex];
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

	size_t imageBaseIndex = size_t(spinImageIndexIndex) * spinImageWidthPixels * spinImageWidthPixels;
    for(size_t i = threadIdx.x; i < spinImageWidthPixels * spinImageWidthPixels; i += blockDim.x) {
        descriptors.content[imageBaseIndex + i] = localSpinImage[i];
    }

}

array<spinImagePixelType> SpinImage::gpu::generateSpinImages(
        DeviceMesh device_mesh,
        float spinImageWidth,
        size_t sampleCount,
        SpinImage::debug::SIRunInfo* runInfo)
{
    auto totalExecutionTimeStart = std::chrono::steady_clock::now();

    size_t descriptorBufferLength = device_mesh.vertexCount * spinImageWidthPixels * spinImageWidthPixels;
	size_t descriptorBufferSize = sizeof(float) * descriptorBufferLength;
	size_t areaArrayLength = device_mesh.vertexCount / 3;
	size_t areaArraySize = areaArrayLength * sizeof(float);
	curandState* device_randomState;
	array<float2> device_coefficients;

	array<spinImagePixelType> device_descriptors;
	array<float> device_areaArray;
	array<float> device_cumulativeAreaArray;
	array<float3> device_pointSamples;

	// -- Initialisation --
	auto initialisationStart = std::chrono::steady_clock::now();

		checkCudaErrors(cudaMalloc(&device_descriptors.content, descriptorBufferSize));
		checkCudaErrors(cudaMalloc(&device_areaArray.content, areaArraySize));
		checkCudaErrors(cudaMalloc(&device_cumulativeAreaArray.content, areaArraySize));
		checkCudaErrors(cudaMalloc(&device_pointSamples.content, sizeof(float3) * sampleCount));
		checkCudaErrors(cudaMalloc(&device_randomState, sizeof(curandState) * (size_t)SAMPLE_COEFFICIENT_THREAD_COUNT));
		checkCudaErrors(cudaMalloc(&device_coefficients.content, sizeof(float2) * sampleCount));

		device_descriptors.length = device_mesh.vertexCount;
		device_areaArray.length = (unsigned) areaArrayLength;
		device_cumulativeAreaArray.length = (unsigned) areaArrayLength;
		device_pointSamples.length = sampleCount;

		CudaLaunchDimensions valueSetSettings = calculateCudaLaunchDimensions(descriptorBufferLength);
		CudaLaunchDimensions areaSettings = calculateCudaLaunchDimensions(device_areaArray.length);
		CudaLaunchDimensions cumulativeAreaSettings = calculateCudaLaunchDimensions(device_areaArray.length);

		setValue <spinImagePixelType><<<valueSetSettings.blocksPerGrid, valueSetSettings.threadsPerBlock >>> (device_descriptors.content, descriptorBufferLength, 0);
		cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());

	std::chrono::milliseconds initialisationDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - initialisationStart);

	// -- Mesh Sampling --
	auto meshSamplingStart = std::chrono::steady_clock::now();

		calculateAreas <<<areaSettings.blocksPerGrid, areaSettings.threadsPerBlock >>> (device_areaArray, device_mesh);
		cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());

		calculateCumulativeAreas<<<cumulativeAreaSettings.blocksPerGrid, cumulativeAreaSettings.threadsPerBlock>>>(device_areaArray, device_cumulativeAreaArray);
		cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());

		generateRandomSampleCoefficients<<<SAMPLE_COEFFICIENT_THREAD_COUNT / 32, 32>>>(device_coefficients, device_randomState, sampleCount);
		cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());

		sampleMesh <<<areaSettings.blocksPerGrid, areaSettings.threadsPerBlock>>>(device_mesh, device_cumulativeAreaArray, device_pointSamples, device_coefficients, sampleCount);
		cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());

	std::chrono::milliseconds meshSamplingDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - meshSamplingStart);

	// -- Spin Image Generation --
	auto generationStart = std::chrono::steady_clock::now();

	    createDescriptors <<<device_mesh.vertexCount, 416>>>(
	            device_mesh,
	            device_pointSamples,
	            device_descriptors,
	            device_cumulativeAreaArray,
	            sampleCount,
	            float(spinImageWidthPixels)/spinImageWidth);
	    checkCudaErrors(cudaDeviceSynchronize());
	    checkCudaErrors(cudaGetLastError());

	std::chrono::milliseconds generationDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - generationStart);

	// -- Cleanup --

	checkCudaErrors(cudaFree(device_areaArray.content));
	checkCudaErrors(cudaFree(device_cumulativeAreaArray.content));
	checkCudaErrors(cudaFree(device_pointSamples.content));
	checkCudaErrors(cudaFree(device_randomState));
	checkCudaErrors(cudaFree(device_coefficients.content));

    std::chrono::milliseconds totalExecutionDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - totalExecutionTimeStart);

    if(runInfo != nullptr) {
	    runInfo->totalExecutionTimeSeconds = double(totalExecutionDuration.count()) / 1000.0;
		runInfo->initialisationTimeSeconds = double(initialisationDuration.count()) / 1000.0;
		runInfo->meshSamplingTimeSeconds = double(meshSamplingDuration.count()) / 1000.0;
	    runInfo->generationTimeSeconds = double(generationDuration.count()) / 1000.0;
	}

	return device_descriptors;
}

