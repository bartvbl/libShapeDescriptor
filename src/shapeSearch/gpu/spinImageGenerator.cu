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

#include <shapeSearch/gpu/types/DeviceMesh.h>
#include <shapeSearch/gpu/types/CudaLaunchDimensions.h>
#include <shapeSearch/gpu/setValue.cuh>

#define SAMPLE_COEFFICIENT_THREAD_COUNT 4096

// Classical spin image generation constants
// Number of threads per warp in classical spin image generation
const int SPIN_IMAGE_GENERATION_WARP_SIZE = 32;

__device__ __inline__ float signedArea(float2 p1, float2 p2, float2 p3)
{
	return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y);
}

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


__device__ __inline__ float calculateScaleFactor(float3 vector1, float3 vector2)
{
	return dot(vector1, vector2) / dot(vector2, vector2);
}

__device__ __inline__ bool isValueEquivalent(float value1, float value2)
{
	return abs(value1 - value2) < 0.001;
}

__device__ __inline__ void lookupTriangleVertices(DeviceMesh mesh, int triangleIndex, float3* triangleVertices) {
	assert(triangleIndex >= 0);
	assert((3 * triangleIndex) + 2 < mesh.indexCount);

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
__global__ void createSampledMesh(DeviceMesh mesh, array<float> areaArray, array<float3> pointSamples, array<float2> coefficients, int sampleCount) {
	int triangleIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if(triangleIndex >= mesh.indexCount / 3)
	{
		return;
	}

	float3 triangleVertices[3];
	lookupTriangleVertices(mesh, triangleIndex, triangleVertices);

	SampleBounds bounds = calculateSampleBounds(areaArray, triangleIndex, sampleCount);

	for(int sample = 0; sample < bounds.sampleCount; sample++) {
		size_t sampleIndex = bounds.sampleStartIndex + sample;

		if(sampleIndex >= sampleCount) {
			printf("Sample %i/%i was skipped.\n", bounds.sampleStartIndex + sample, bounds.sampleCount);
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
__global__ void createDescriptors(DeviceMesh mesh, array<float3> pointSamples, array<classicSpinImagePixelType> descriptors, array<float> areaArray, int sampleCount, float spinImageWidth)
{
	int spinImageIndexIndex = blockIdx.x;

	if(spinImageIndexIndex >= mesh.indexCount)
	{
		return;
	}

	int spinImageIndex = spinImageIndexIndex;

	float3 vertex;
	float3 normal;

	vertex.x = mesh.vertices_x[spinImageIndexIndex];
	vertex.y = mesh.vertices_y[spinImageIndexIndex];
	vertex.z = mesh.vertices_z[spinImageIndexIndex];

	normal.x = mesh.normals_x[spinImageIndexIndex];
	normal.y = mesh.normals_y[spinImageIndexIndex];
	normal.z = mesh.normals_z[spinImageIndexIndex];

	for (int triangleIndex = threadIdx.x; triangleIndex < mesh.indexCount / 3; triangleIndex += SPIN_IMAGE_GENERATION_WARP_SIZE)
	{
		SampleBounds bounds = calculateSampleBounds(areaArray, triangleIndex, sampleCount);

		for(unsigned int sample = 0; sample < bounds.sampleCount; sample++)
		{
			size_t sampleIndex = bounds.sampleStartIndex + sample;

			if(sampleIndex >= sampleCount) {
				printf("Sample %i/%i was skipped.\n", sampleIndex, bounds.sampleCount);
				continue;
			}

			assert(sampleIndex < pointSamples.length);

			float3 samplePoint = pointSamples.content[sampleIndex];
			float2 sampleAlphaBeta = calculateAlphaBeta(vertex, normal, samplePoint);

			float floatSpinImageCoordinateX = (sampleAlphaBeta.x / spinImageWidth);
			float floatSpinImageCoordinateY = (sampleAlphaBeta.y / spinImageWidth);

			int baseSpinImageCoordinateX = (int) floorf(floatSpinImageCoordinateX);
			int baseSpinImageCoordinateY = (int) floorf(floatSpinImageCoordinateY);

			float interPixelX = floatSpinImageCoordinateX - floorf(floatSpinImageCoordinateX);
			float interPixelY = floatSpinImageCoordinateY - floorf(floatSpinImageCoordinateY);

			const unsigned int halfSpinImageSizePixels = spinImageWidthPixels / 2;

            if (baseSpinImageCoordinateX + 0 >= 0 &&
                baseSpinImageCoordinateX + 0 < spinImageWidthPixels &&
                baseSpinImageCoordinateY + 0 >= -halfSpinImageSizePixels &&
                baseSpinImageCoordinateY + 0 < halfSpinImageSizePixels)
            {
                size_t valueIndex = size_t(spinImageIndex) * spinImageWidthPixels * spinImageWidthPixels +
                                 (baseSpinImageCoordinateY + 0 + spinImageWidthPixels / 2) * spinImageWidthPixels +
                                 baseSpinImageCoordinateX + 0;
                atomicAdd(&(descriptors.content[valueIndex]), (interPixelX) * (interPixelY));
            }

            if (baseSpinImageCoordinateX + 1 >= 0 &&
                baseSpinImageCoordinateX + 1 < spinImageWidthPixels &&
                baseSpinImageCoordinateY + 0 >= -halfSpinImageSizePixels &&
                baseSpinImageCoordinateY + 0 < halfSpinImageSizePixels)
            {
                size_t valueIndex = size_t(spinImageIndex) * spinImageWidthPixels * spinImageWidthPixels +
                                    (baseSpinImageCoordinateY + 0 + spinImageWidthPixels / 2) * spinImageWidthPixels +
                                    baseSpinImageCoordinateX + 1;
                atomicAdd(&(descriptors.content[valueIndex]), (1.0f - interPixelX) * (interPixelY));
            }

            if (baseSpinImageCoordinateX + 1 >= 0 &&
                baseSpinImageCoordinateX + 1 < spinImageWidthPixels &&
                baseSpinImageCoordinateY + 1 >= -halfSpinImageSizePixels &&
                baseSpinImageCoordinateY + 1 < halfSpinImageSizePixels)
            {
                size_t valueIndex = size_t(spinImageIndex) * spinImageWidthPixels * spinImageWidthPixels +
                                    (baseSpinImageCoordinateY + 1 + spinImageWidthPixels / 2) * spinImageWidthPixels +
                                    baseSpinImageCoordinateX + 1;
                atomicAdd(&(descriptors.content[valueIndex]), (1.0f - interPixelX) * (1.0f - interPixelY));
            }

            if (baseSpinImageCoordinateX + 0 >= 0 &&
                baseSpinImageCoordinateX + 0 < spinImageWidthPixels &&
                baseSpinImageCoordinateY + 1 >= -halfSpinImageSizePixels &&
                baseSpinImageCoordinateY + 1 < halfSpinImageSizePixels)
            {
                size_t valueIndex = size_t(spinImageIndex) * spinImageWidthPixels * spinImageWidthPixels +
                                    (baseSpinImageCoordinateY + 1 + spinImageWidthPixels / 2) * spinImageWidthPixels +
                                    baseSpinImageCoordinateX + 0;
                atomicAdd(&(descriptors.content[valueIndex]), (interPixelX) * (1.0f - interPixelY));
            }
		}
	}
}


array<classicSpinImagePixelType> createClassicDescriptors(DeviceMesh device_mesh, cudaDeviceProp device_information, size_t sampleCount)
{
	// In principle, these kernels should only be run once per vertex.
	// However, since we also need a normal, and the same vertex can have different normals in different situations,
	// we need to run the vertex index multiple times to ensure we create a spin image for every case.
	// This is unfortunately very much overkill, but I currently don't know how to fix it.

	size_t descriptorBufferLength = device_mesh.vertexCount * spinImageWidthPixels * spinImageWidthPixels;
	size_t descriptorBufferSize = sizeof(float) * descriptorBufferLength;
	array<classicSpinImagePixelType> device_descriptors;
	checkCudaErrors(cudaMalloc(&device_descriptors.content, descriptorBufferSize));
	device_descriptors.length = device_mesh.vertexCount;
	std::cout << "\t- Allocating descriptor array (size: " << descriptorBufferSize << ", pointer: " << device_descriptors.content << ")" << std::endl;

	array<float> device_areaArray;
	array<float> device_cumulativeAreaArray;

	// Calculate triangle count
	size_t areaArrayLength = device_mesh.indexCount / 3;

	size_t areaArraySize = areaArrayLength * sizeof(float);
	checkCudaErrors(cudaMalloc(&device_areaArray.content, areaArraySize));
	checkCudaErrors(cudaMalloc(&device_cumulativeAreaArray.content, areaArraySize));
	device_areaArray.length = (unsigned) areaArrayLength;
	device_cumulativeAreaArray.length = (unsigned) areaArrayLength;

	std::cout << "\t- Initialising descriptor array" << std::endl;
	CudaLaunchDimensions valueSetSettings = calculateCudaLaunchDimensions(descriptorBufferLength, device_information);
	setValue <classicSpinImagePixelType><<<valueSetSettings.blocksPerGrid, valueSetSettings.threadsPerBlock >>> (device_descriptors.content, descriptorBufferLength, 0);

	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	std::cout << "\t- Calculating areas" << std::endl;
    CudaLaunchDimensions areaSettings = calculateCudaLaunchDimensions(device_areaArray.length, device_information);
	calculateAreas <<<areaSettings.blocksPerGrid, areaSettings.threadsPerBlock >>> (device_areaArray, device_mesh);

	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	

	std::cout << "\t- Calculating sample values" << std::endl;
    CudaLaunchDimensions cumulativeAreaSettings = calculateCudaLaunchDimensions(device_areaArray.length,
																				device_information);
	calculateCumulativeAreas<<<cumulativeAreaSettings.blocksPerGrid, cumulativeAreaSettings.threadsPerBlock>>>(device_areaArray, device_cumulativeAreaArray);
	//shuffle_prefix_scan_float(device_areaArray.content, device_cumulativeAreaArray.content, device_areaArray.length);

	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	float cumulativeArea;

	checkCudaErrors(cudaMemcpy(&cumulativeArea, device_cumulativeAreaArray.content + areaArrayLength - 1, sizeof(float), cudaMemcpyDeviceToHost));
	std::cout << "\t- Cumulative Area: " << cumulativeArea << std::endl;

	std::cout << "\t- Calculating random coefficients" << std::endl;

	curandState* device_randomState;
	checkCudaErrors(cudaMalloc(&device_randomState, sizeof(curandState) * (size_t)SAMPLE_COEFFICIENT_THREAD_COUNT));

    CudaLaunchDimensions sampleSettings = calculateCudaLaunchDimensions(SAMPLE_COEFFICIENT_THREAD_COUNT,
																		device_information);

	array<float2> device_coefficients;
	checkCudaErrors(cudaMalloc(&device_coefficients.content, sizeof(float2) * sampleCount));

	std::cout << "\t- Sampling input model using " << sampleCount << " samples." << std::endl;
	auto sampleStart = std::chrono::steady_clock::now();

	generateRandomSampleCoefficients<<<SAMPLE_COEFFICIENT_THREAD_COUNT / 32, 32>>>(device_coefficients, device_randomState, sampleCount);

	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	array<float3> device_pointSamples;
	checkCudaErrors(cudaMalloc(&device_pointSamples.content, sizeof(float3) * sampleCount));
	device_pointSamples.length = sampleCount;

	createSampledMesh<<<areaSettings.blocksPerGrid, areaSettings.threadsPerBlock>>>(device_mesh, device_cumulativeAreaArray, device_pointSamples, device_coefficients, sampleCount);

	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	std::chrono::milliseconds sampleDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - sampleStart);
	std::cout << "Execution time:" << sampleDuration.count() << std::endl;

	dim3 blockSizes;
	blockSizes.x = device_mesh.vertexCount; // Run one 3x3x3 area for each image
	blockSizes.y = 27; // 3 x 3 x 3 area around the cube containing the vertex
	blockSizes.z = 1;  // Just a single dimension.

	auto start = std::chrono::steady_clock::now();

	std::cout << "\t- Running spin image kernel" << std::endl;
	createDescriptors <<<blockSizes, SPIN_IMAGE_GENERATION_WARP_SIZE >>>(device_mesh, device_pointSamples, device_descriptors, device_cumulativeAreaArray, sampleCount, 1.0f);

	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	std::chrono::milliseconds duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
	std::cout << "Execution time:" << duration.count() << std::endl;

	std::cout << "\t- Copying results to CPU" << std::endl;

	array<classicSpinImagePixelType> host_descriptors;
	host_descriptors.content = new classicSpinImagePixelType[descriptorBufferLength];
	host_descriptors.length = device_descriptors.length;

	checkCudaErrors(cudaMemcpy(host_descriptors.content, device_descriptors.content, descriptorBufferSize, cudaMemcpyDeviceToHost));

	return device_descriptors;
}

