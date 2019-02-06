#include "spinImageGenerator.cuh"

#include <chrono>

#include "cuda_runtime.h"

#include "cudaCommon.h"
#include <assert.h>
#include "deviceMesh.h"
#include "device_launch_parameters.h"
#include "nvidia/shfl_scan.cuh"
#include "nvidia/helper_math.h"

#include "nvidia/helper_cuda.h"
#include <iostream>
#include <curand.h>
#include <curand_kernel.h>
#include <map>

#define SAMPLE_COEFFICIENT_THREAD_COUNT 4096

__device__ __inline__ float signedArea(float2 p1, float2 p2, float2 p3)
{
	return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y);
}

__device__ __inline__ float2 calculateAlphaBeta(float3 spinVertex, float3 spinNormal, float3 point, CubePartition partition)
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

	//printf("(%f, %f, %f) + (%f, %f, %f) + (%f, %f, %f) -> %f, %f\n", spinVertex.x, spinVertex.y, spinVertex.z, spinNormal.x, spinNormal.y, spinNormal.z, point.x, point.y, point.z, alpha, beta);
	return alphabeta;
}

__device__ __inline__ float3 transformCoordinate(float3 vertex, float3 spin_vertex, float3 normal)
{


	float2 sineCosineAlpha = normalize(make_float2(normal.x, normal.y));

	bool is_n_a_not_zero = !((abs(normal.x) < 0.0001f) && (abs(normal.y) < 0.0001f));

	PrecalculatedSettings settings;

	if (is_n_a_not_zero)
	{
		settings.alignmentProjection_n_ax = sineCosineAlpha.x;
		settings.alignmentProjection_n_ay = sineCosineAlpha.y;
	}
	else
	{
		// Leave values unchanged
		settings.alignmentProjection_n_ax = 1;
		settings.alignmentProjection_n_ay = 0;
	}

	float transformedNormalX = settings.alignmentProjection_n_ax * normal.x + settings.alignmentProjection_n_ay * normal.y;

	float2 sineCosineBeta = normalize(make_float2(transformedNormalX, normal.z));

	bool is_n_b_not_zero = !((abs(transformedNormalX) < 0.0001f) && (abs(normal.z) < 0.0001f));

	if (is_n_b_not_zero)
	{
		settings.alignmentProjection_n_bx = sineCosineBeta.x;
		settings.alignmentProjection_n_bz = sineCosineBeta.y; // discrepancy between axis here is because we are using a 2D vector on 3D axis.
	}
	else
	{
		// Leave input values unchanged
		settings.alignmentProjection_n_bx = 1;
		settings.alignmentProjection_n_bz = 0;
	}

	float3 transformedCoordinate = vertex - spin_vertex;

	float initialTransformedX = transformedCoordinate.x;
	transformedCoordinate.x = settings.alignmentProjection_n_ax * transformedCoordinate.x + settings.alignmentProjection_n_ay * transformedCoordinate.y;
	transformedCoordinate.y = -settings.alignmentProjection_n_ay * initialTransformedX + settings.alignmentProjection_n_ax * transformedCoordinate.y;

	// Order matters here
	initialTransformedX = transformedCoordinate.x;
	transformedCoordinate.x = settings.alignmentProjection_n_bz * transformedCoordinate.x - settings.alignmentProjection_n_bx * transformedCoordinate.z;
	transformedCoordinate.z = settings.alignmentProjection_n_bx * initialTransformedX + settings.alignmentProjection_n_bz * transformedCoordinate.z;

	return transformedCoordinate;
}

__device__ __inline__ float calculateScaleFactor(float3 vector1, float3 vector2)
{
	return dot(vector1, vector2) / dot(vector2, vector2);
}

__device__ __inline__ bool isValueEquivalent(float value1, float value2)
{
	return abs(value1 - value2) < 0.001;
}

// @Assumption: vertex is measured relative to the origin of the grid
__device__ __inline__ int3 calculateCubeLocation(CubePartition cubePartition, float3 vertex)
{
	// To add the vertex to the appropriate cube, we first calculate the cube the vertex belongs to.
	int3 cubeLocation;

	cubeLocation.x = int(vertex.x / cubePartition.cubeSize);
	cubeLocation.y = int(vertex.y / cubePartition.cubeSize);
	cubeLocation.z = int(vertex.z / cubePartition.cubeSize);

	return cubeLocation;
}

__device__ __inline__ int calculateCubeIndex(CubePartition cubePartition, int3 cubeLocation)
{
	return (cubePartition.cubeCounts.x * cubePartition.cubeCounts.y * cubeLocation.z) +
		(cubePartition.cubeCounts.x * cubeLocation.y) +
		(cubeLocation.x);
}

__device__ __inline__ bool isInsideCube(float3 vertex, float3 boundingBoxMin, float3 boundingBoxMax)
{
	return
		vertex.x >= boundingBoxMin.x && vertex.x <= boundingBoxMax.x &&
		vertex.y >= boundingBoxMin.y && vertex.y <= boundingBoxMax.y &&
		vertex.z >= boundingBoxMin.z && vertex.z <= boundingBoxMax.z;
}

__device__ __inline__ void lookupTriangleVertices(Mesh mesh, CubePartition partition, int triangleIndex, float3* triangleVertices) {
	int vertexIndices[3];

	//printf("%i -> %i\n", triangleIndex, triangleIndex);

	assert(triangleIndex >= 0);
	assert((3 * triangleIndex) + 2 < mesh.indexCount);

	vertexIndices[0] = mesh.indices_vertex0[triangleIndex];
	vertexIndices[1] = mesh.indices_vertex1[triangleIndex];
	vertexIndices[2] = mesh.indices_vertex2[triangleIndex];

	assert(vertexIndices[0] >= 0);
	assert(vertexIndices[0] < mesh.vertexCount);
	assert(vertexIndices[1] >= 0);
	assert(vertexIndices[1] < mesh.vertexCount);
	assert(vertexIndices[2] >= 0);
	assert(vertexIndices[2] < mesh.vertexCount);

	triangleVertices[0].x = mesh.vertices_x[vertexIndices[0]];
	triangleVertices[0].y = mesh.vertices_y[vertexIndices[0]];
	triangleVertices[0].z = mesh.vertices_z[vertexIndices[0]];

	triangleVertices[1].x = mesh.vertices_x[vertexIndices[1]];
	triangleVertices[1].y = mesh.vertices_y[vertexIndices[1]];
	triangleVertices[1].z = mesh.vertices_z[vertexIndices[1]];

	triangleVertices[2].x = mesh.vertices_x[vertexIndices[2]];
	triangleVertices[2].y = mesh.vertices_y[vertexIndices[2]];
	triangleVertices[2].z = mesh.vertices_z[vertexIndices[2]];

	assert(!isnan(triangleVertices[0].x) && !isnan(triangleVertices[0].y) && !isnan(triangleVertices[0].z));
	assert(!isnan(triangleVertices[1].x) && !isnan(triangleVertices[1].y) && !isnan(triangleVertices[1].z));
	assert(!isnan(triangleVertices[2].x) && !isnan(triangleVertices[2].y) && !isnan(triangleVertices[2].z));

	// Converting the coordinates from mesh space to cube grid space.
	triangleVertices[0] -= mesh.boundingBoxMin;
	triangleVertices[1] -= mesh.boundingBoxMin;
	triangleVertices[2] -= mesh.boundingBoxMin;
}

typedef struct SampleBounds {
	float sampleStart;
	size_t sampleCount;
	float areaStart;
	float areaEnd;
	float triangleArea;
	size_t sampleStartIndex;
} SampleBounds;

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

	sampleBounds.triangleArea = sampleBounds.areaEnd - sampleBounds.areaStart;

	size_t firstIndexInRange = (size_t) (sampleBounds.areaStart / areaStepSize) + 1;
	size_t lastIndexInRange = (size_t) (sampleBounds.areaEnd / areaStepSize);

	float remainder = fmod(sampleBounds.areaStart, areaStepSize);
	sampleBounds.sampleCount = lastIndexInRange - firstIndexInRange + 1; // Offset is needed to ensure bounds are correct
	sampleBounds.sampleStart = sampleBounds.areaStart + remainder;
	sampleBounds.sampleStartIndex = firstIndexInRange - 1;

	/*printf("Area: %i: %f, %f -> %f, %f -> %i, %i -> %f, %i, %f, %i\n",
		   triangleIndex, maxArea, areaStepSize,
		   sampleBounds.areaStart, sampleBounds.triangleArea,
		   firstIndexInRange, lastIndexInRange,
		   remainder, sampleBounds.sampleCount, sampleBounds.sampleStart, sampleBounds.sampleStartIndex);*/

	return sampleBounds;
}

// One thread = One triangle
__global__ void calculateAreas(floatArray areaArray, Mesh mesh, CubePartition partition)
{
	int triangleIndex = blockDim.x * blockIdx.x + threadIdx.x;
	if (triangleIndex >= areaArray.length)
	{
		return;
	}
	float3 vertices[3];
	lookupTriangleVertices(mesh, partition, triangleIndex, vertices);
	float3 v1 = vertices[1] - vertices[0];
	float3 v2 = vertices[2] - vertices[0];
	float area = length(cross(v1, v2)) / 2.0;
	areaArray.content[triangleIndex] = area;
	//printf("%i -> %f\n", triangleIndex, area);
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
__global__ void createSampledMesh(Mesh mesh, CubePartition partition, array<float> areaArray, array<float3> pointSamples, array<float2> coefficients, int sampleCount) {
	int triangleIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if(triangleIndex >= mesh.indexCount / 3)
	{
		return;
	}

	//printf("%i -> %f\n", triangleIndex, areaArray.content[triangleIndex]);

	float3 triangleVertices[3];
	lookupTriangleVertices(mesh, partition, triangleIndex, triangleVertices);

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


		assert(sampleIndex >= 0);
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


__global__ void createDescriptors(Mesh mesh, CubePartition partition, array<float3> pointSamples, array<classicSpinImagePixelType> descriptors, array<float> areaArray, int sampleCount)
{
	int spinImageIndexIndex = blockIdx.x;
	int rawThreadIndex = threadIdx.x+blockDim.x*blockIdx.x;

	if(spinImageIndexIndex >= mesh.indexCount)
	{
		return;
	}

	float maxArea = areaArray.content[areaArray.length - 1];
	float areaStepSize = maxArea / (float)sampleCount;

	float physicalSpinPixelWidth = (partition.cubeSize) / (2.0f * (float)(spinImageWidthPixels)); // Half cubesize optimisation

	int vertexIndexIndex = spinImageIndexIndex;
	int spinImageIndex = spinImageIndexIndex;


	float3 vertex;
	float3 normal;

	vertex.x = mesh.vertices_x[spinImageIndexIndex];
	vertex.y = mesh.vertices_y[spinImageIndexIndex];
	vertex.z = mesh.vertices_z[spinImageIndexIndex];

	normal.x = mesh.normals_x[spinImageIndexIndex];
	normal.y = mesh.normals_y[spinImageIndexIndex];
	normal.z = mesh.normals_z[spinImageIndexIndex];

	// calculate coordinate relative to mesh origin
	// the coordinates can otherwise not be compared to cube coordinates
	vertex -= mesh.boundingBoxMin;

	int3 cubeLocation = calculateCubeLocation(partition, vertex);

	int3 blockDeltaCoordinate;
	blockDeltaCoordinate.x = (blockIdx.y / 9);
	blockDeltaCoordinate.y = ((blockIdx.y - (9 * blockDeltaCoordinate.x)) / 3);
	blockDeltaCoordinate.z = ((blockIdx.y - (9 * blockDeltaCoordinate.x) - (3 * blockDeltaCoordinate.y)));

	blockDeltaCoordinate.x -= 1;
	blockDeltaCoordinate.y -= 1;
	blockDeltaCoordinate.z -= 1;


	int3 currentCubeLocation = cubeLocation + blockDeltaCoordinate;

	if( currentCubeLocation.x < 0 || currentCubeLocation.y < 0 || currentCubeLocation.z < 0 ||
		currentCubeLocation.x >= partition.cubeCounts.x || currentCubeLocation.y >= partition.cubeCounts.y || currentCubeLocation.z >= partition.cubeCounts.z)
	{
		return;
	}

	int currentCubeIndex = calculateCubeIndex(partition, currentCubeLocation);

	int cubeContentStartIndex = partition.startIndices[currentCubeIndex];
	int cubeContentLength = partition.lengths[currentCubeIndex];
	int cubeContentEndIndex = cubeContentStartIndex + cubeContentLength;


	for (int triangleIndexIndex = cubeContentStartIndex + threadIdx.x; triangleIndexIndex < cubeContentEndIndex; triangleIndexIndex+=SPIN_IMAGE_GENERATION_WARP_SIZE)
	{


		// The main cube should include all triangles
		// We can thus skip searching for a valid triangle


		unsigned int triangleIndex = partition.duplicated_triangle_indices[triangleIndexIndex];
		unsigned int minCubeIndex = partition.minCubeIndices.content[triangleIndex];
		bool isTriangleInLowestCube = minCubeIndex == currentCubeIndex;


		if (!isTriangleInLowestCube) {
			continue;
		}



		// START OF SAMPLING

		SampleBounds bounds = calculateSampleBounds(areaArray, triangleIndex, sampleCount);

		for(unsigned int sample = 0; sample < bounds.sampleCount; sample++)
		{
			size_t sampleIndex = bounds.sampleStartIndex + sample;

			if(sampleIndex >= sampleCount) {
				printf("Sample %i/%i was skipped.\n", sampleIndex, bounds.sampleCount);
				continue;
			}

			assert(sampleIndex < pointSamples.length);
			assert(sampleIndex >= 0);

			float3 samplePoint = pointSamples.content[sampleIndex];
			float2 sampleAlphaBeta = calculateAlphaBeta(vertex, normal, samplePoint, partition);

			float floatSpinImageCoordinateX = (sampleAlphaBeta.x / physicalSpinPixelWidth);
			float floatSpinImageCoordinateY = (sampleAlphaBeta.y / physicalSpinPixelWidth);

			int baseSpinImageCoordinateX = (int) floorf(floatSpinImageCoordinateX);
			int baseSpinImageCoordinateY = (int) floorf(floatSpinImageCoordinateY);

			float interPixelX = floatSpinImageCoordinateX - floorf(floatSpinImageCoordinateX);
			float interPixelY = floatSpinImageCoordinateY - floorf(floatSpinImageCoordinateY);

			//if(!std::isnan(interPixelX) && !std::isnan(interPixelY)) {
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
//            } else {
//                printf("NaN detected!\n"
//                    "\tInterPixelX: %f\n"
//                    "\tfloatSpinImageCoordinateX: %f\n"
//                    "\tAlphabeta: (%f, %f)\n"
//                    "\tSpin normal: (%f, %f, %f)\n", interPixelX, floatSpinImageCoordinateX, sampleAlphaBeta.x, sampleAlphaBeta.y, normal.x, normal.y, normal.z);
		  //  }
		}
	}
}



VertexDescriptors createClassicDescriptors(Mesh device_mesh, CubePartition device_cubePartition, cudaDeviceProp device_information, OutputImageSettings imageSettings, size_t sampleCount)
{
	// In principle, these kernels should only be run once per vertex.
	// However, since we also need a normal, and the same vertex can have different normals in different situations,
	// we need to run the vertex index multiple times to ensure we create a spin image for every case.
	// This is unfortunately very much overkill, but I currently don't know how to fix it.

	size_t imageCount = device_mesh.vertexCount;

	size_t descriptorBufferLength = device_mesh.vertexCount * spinImageWidthPixels * spinImageWidthPixels;
	size_t descriptorBufferSize = sizeof(float) * descriptorBufferLength;
	VertexDescriptors device_descriptors;
	checkCudaErrors(cudaMalloc(&device_descriptors.classicDescriptorArray.content, descriptorBufferSize));
	device_descriptors.classicDescriptorArray.length = device_mesh.vertexCount;
	std::cout << "\t- Allocating descriptor array (size: " << descriptorBufferSize << ", pointer: " << device_descriptors.classicDescriptorArray.content << ")" << std::endl;

	device_descriptors.isNew = false;
	device_descriptors.isClassic = true;

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
	CudaSettings valueSetSettings = calculateCUDASettings(descriptorBufferLength, device_information);
	setValue <classicSpinImagePixelType><<<valueSetSettings.blocksPerGrid, valueSetSettings.threadsPerBlock >>> (device_descriptors.classicDescriptorArray.content, descriptorBufferLength, 0);

	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	std::cout << "\t- Calculating areas" << std::endl;
	CudaSettings areaSettings = calculateCUDASettings(device_areaArray.length, device_information);
	calculateAreas <<<areaSettings.blocksPerGrid, areaSettings.threadsPerBlock >>> (device_areaArray, device_mesh, device_cubePartition);

	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	

	std::cout << "\t- Calculating sample values" << std::endl;
	CudaSettings cumulativeAreaSettings = calculateCUDASettings(device_areaArray.length, device_information);
	calculateCumulativeAreas<<<cumulativeAreaSettings.blocksPerGrid, cumulativeAreaSettings.threadsPerBlock>>>(device_areaArray, device_cumulativeAreaArray);
	//shuffle_prefix_scan_float(device_areaArray.content, device_cumulativeAreaArray.content, device_areaArray.length);

	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	float cumulativeArea;

	checkCudaErrors(cudaMemcpy(&cumulativeArea, device_cumulativeAreaArray.content + areaArrayLength - 1, sizeof(float), cudaMemcpyDeviceToHost));
	std::cout << "\t- Cumulative Area: " << cumulativeArea << std::endl;

	// PRESSING THE OVERRIDE BUTTON! 
	//sampleCount = 2 * size_t(cumulativeArea);

	/*float* host_cumulativeAreas = new float[areaArrayLength];
	checkCudaErrors(cudaMemcpy(host_cumulativeAreas, device_cumulativeAreaArray.content, areaArraySize, cudaMemcpyDeviceToHost));
	for(int i = 0; i < areaArrayLength; i++) {
		std::cout << host_cumulativeAreas[i] << ", "; 
	}*/

	std::cout << "\t- Calculating random coefficients" << std::endl;

	curandState* device_randomState;
	checkCudaErrors(cudaMalloc(&device_randomState, sizeof(curandState) * (size_t)SAMPLE_COEFFICIENT_THREAD_COUNT));

	CudaSettings sampleSettings = calculateCUDASettings(SAMPLE_COEFFICIENT_THREAD_COUNT, device_information);

	array<float2> device_coefficients;
	checkCudaErrors(cudaMalloc(&device_coefficients.content, sizeof(float2) * sampleCount));

	std::cout << "\t- Sampling input model using " << sampleCount << " samples." << std::endl;
	auto sampleStart = std::chrono::steady_clock::now();

	generateRandomSampleCoefficients<<<SAMPLE_COEFFICIENT_THREAD_COUNT / 32, 32>>>(device_coefficients, device_randomState, sampleCount);

	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	//float2* localCoefficients = new float2[sampleCount];
	//checkCudaErrors(cudaMemcpy(localCoefficients, device_coefficients.content, sizeof(float2) * sampleCount, cudaMemcpyDeviceToHost));
	//std::ofstream sampleValueStream("../output/sample_values.txt");
	//for(int i = 0; i < sampleCount; i++) {
	//    sampleValueStream << localCoefficients[i].x << ", " << localCoefficients[i].y << std::endl;
	//}

	//sampleValueStream.close();

	array<float3> device_pointSamples;
	checkCudaErrors(cudaMalloc(&device_pointSamples.content, sizeof(float3) * sampleCount));
	device_pointSamples.length = sampleCount;

	

	createSampledMesh<<<areaSettings.blocksPerGrid, areaSettings.threadsPerBlock>>>(device_mesh, device_cubePartition, device_cumulativeAreaArray, device_pointSamples, device_coefficients, sampleCount);

	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	std::chrono::milliseconds sampleDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - sampleStart);
	std::cout << "Execution time:" << sampleDuration.count() << std::endl;


	/*float3* pointSamples = new float3[sampleCount];
	checkCudaErrors(cudaMemcpy(pointSamples, device_pointSamples.content, sizeof(float3) * sampleCount, cudaMemcpyDeviceToHost));

	std::ofstream sampleFileStream("../output/pointSamples.txt");
	for(int i = 0; i < sampleCount; i++) {
		sampleFileStream << pointSamples[i].x << " " << pointSamples[i].y << " " << pointSamples[i].z << std::endl;
	}
	sampleFileStream.close();*/

	dim3 blockSizes;
	blockSizes.x = device_mesh.vertexCount; // Run one 3x3x3 area for each image
	blockSizes.y = 27; // 3 x 3 x 3 area around the cube containing the vertex
	blockSizes.z = 1;  // Just a single dimension.

	auto start = std::chrono::steady_clock::now();

	std::cout << "\t- Running spin image kernel" << std::endl;
	createDescriptors <<<blockSizes, SPIN_IMAGE_GENERATION_WARP_SIZE >>>(device_mesh, device_cubePartition, device_pointSamples, device_descriptors.classicDescriptorArray, device_cumulativeAreaArray, sampleCount);

	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	std::chrono::milliseconds duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
	std::cout << "Execution time:" << duration.count() << std::endl;

	std::cout << "\t- Copying results to CPU" << std::endl;
	VertexDescriptors host_descriptors;
	host_descriptors.classicDescriptorArray.content = new classicSpinImagePixelType[descriptorBufferLength];
	host_descriptors.classicDescriptorArray.length = device_descriptors.classicDescriptorArray.length;
	host_descriptors.isClassic = true;
	checkCudaErrors(cudaMemcpy(host_descriptors.classicDescriptorArray.content, device_descriptors.classicDescriptorArray.content, descriptorBufferSize, cudaMemcpyDeviceToHost));


/*
	std::map<int, int> histogramMap;
	for(size_t i = 0; i < descriptorBufferLength / sizeof(float); i++) {
		float bufferValue = host_descriptors.classicDescriptorArray.content[i];
		int intValue = int(bufferValue);
		if(histogramMap.count(intValue) == 0) {
			histogramMap[intValue] = 0;
		}
		histogramMap[intValue]++;
	}

	std::ofstream histogramFile("../output/projectionCountHistogram.txt");

	for (std::map<int, int>::iterator it = histogramMap.begin(); it != histogramMap.end(); ++it) {
		histogramFile << it->first << " - " << it->second << std::endl;
	}

	histogramFile.close();
*/



	std::ofstream executionTimeFileStream("../output/execution_time.txt");
	executionTimeFileStream << imageCount << std::endl;
	executionTimeFileStream << device_mesh.vertexCount << std::endl;
	executionTimeFileStream << device_mesh.indexCount << std::endl;
	executionTimeFileStream << duration.count() << std::endl;
	executionTimeFileStream << sampleCount << std::endl;
	executionTimeFileStream << int(cumulativeArea) << std::endl;
	executionTimeFileStream << device_cubePartition.cubeCounts.x << std::endl;
	executionTimeFileStream << device_cubePartition.cubeCounts.y << std::endl;
	executionTimeFileStream << device_cubePartition.cubeCounts.z << std::endl;
	executionTimeFileStream.close();

	if(imageSettings.enableOutputImage) {
		std::cout << "\t- Dumping images.." << std::endl;
		dumpImages(host_descriptors, imageSettings);
	}

	return device_descriptors;
}

