#include "quasiSpinImageGenerator.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <shapeSearch/gpu/types/DeviceMesh.h>
#include <shapeSearch/gpu/types/GPURasterisationSettings.h>
#include <shapeSearch/gpu/types/CudaLaunchDimensions.h>
#include <shapeSearch/gpu/setValue.cuh>
#include <shapeSearch/libraryBuildSettings.h>
#include <shapeSearch/common/types/QSIPrecalculatedSettings.h>

#include "nvidia/shfl_scan.cuh"
#include "nvidia/helper_math.h"
#include "nvidia/helper_cuda.h"

#include <assert.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <sstream>

#if QSI_PIXEL_DATATYPE == DATATYPE_UNSIGNED_SHORT
const int SHORT_SINGLE_BOTH_MASK = 0x00010001;
const int SHORT_SINGLE_ONE_MASK = 0x00000001;
const int SHORT_SINGLE_FIRST_MASK = 0x00010000;

const int SHORT_DOUBLE_BOTH_MASK = 0x00020002;
const int SHORT_DOUBLE_ONE_MASK = 0x00000002;
const int SHORT_DOUBLE_FIRST_MASK = 0x00020000;
#endif

const int RASTERISATION_WARP_SIZE = 1024;

__device__ __inline__ QSIPrecalculatedSettings calculateRotationSettings(float3 spinImageNormal);

__device__ __inline__ float transformNormalX(QSIPrecalculatedSettings pre_settings, float3 spinImageNormal)
{
	return pre_settings.alignmentProjection_n_ax * spinImageNormal.x + pre_settings.alignmentProjection_n_ay * spinImageNormal.y;
}

__device__ __inline__ float3 transformCoordinate(float3 vertex, GPURasterisationSettings settings)
{
	QSIPrecalculatedSettings spinImageSettings = calculateRotationSettings(settings.spinImageNormal);

	float3 transformedCoordinate = vertex - settings.spinImageVertex;

	float initialTransformedX = transformedCoordinate.x;
	transformedCoordinate.x = spinImageSettings.alignmentProjection_n_ax * transformedCoordinate.x + spinImageSettings.alignmentProjection_n_ay * transformedCoordinate.y;
	transformedCoordinate.y = -spinImageSettings.alignmentProjection_n_ay * initialTransformedX + spinImageSettings.alignmentProjection_n_ax * transformedCoordinate.y;

	// Order matters here
	initialTransformedX = transformedCoordinate.x;
	transformedCoordinate.x = spinImageSettings.alignmentProjection_n_bz * transformedCoordinate.x - spinImageSettings.alignmentProjection_n_bx * transformedCoordinate.z;
	transformedCoordinate.z = spinImageSettings.alignmentProjection_n_bx * initialTransformedX + spinImageSettings.alignmentProjection_n_bz * transformedCoordinate.z;

	return transformedCoordinate;
}

__device__ __inline__ float2 alignWithPositiveX(float2 midLineDirection, float2 vertex)
{
	float2 transformed;
	transformed.x = midLineDirection.x * vertex.x + midLineDirection.y * vertex.y;
	transformed.y = -midLineDirection.y * vertex.x + midLineDirection.x * vertex.y;
	return transformed;
}

__device__ __inline__ float calculateTransformedZCoordinate(GPURasterisationSettings settings, float3 vertex)
{
	QSIPrecalculatedSettings pre_settings = calculateRotationSettings(settings.spinImageNormal);

	// Translate to origin
	vertex -= settings.spinImageVertex;

	// Since we're looking at 2 axis of a 3D normal vector, both axis may be 0.
	// In this first case: if x = 0 and y = 0, z = 1 or z = -1. This is already part of the vertical plane we'd like the rotated
	// coordinates to be in, so nothing needs to be done.
	float transformedX = pre_settings.alignmentProjection_n_ax * vertex.x + pre_settings.alignmentProjection_n_ay * vertex.y;

	// In this second case: if x = 0 and z = 0, y = 1 or y = -1. In both cases the previous step already rotated the vector to the correct
	// direction, and nothing else needs to be done.
	//float transformedZ = settings.alignmentProjection_n_bx * transformedX + settings.alignmentProjection_n_bz * triangleVertices[i].z;
	float transformedZ = pre_settings.alignmentProjection_n_bx * transformedX + pre_settings.alignmentProjection_n_bz * vertex.z;

	// Account for that samples of pixels are considered centred. We thus need to add a distance value to each z-coordinate.
	//transformedZ -= settings.nudgeDistance;

	return transformedZ;
}

__device__ __inline__ void rasteriseRow(int pixelBaseIndex, newSpinImagePixelType* descriptorArray, unsigned int pixelStart, unsigned int pixelEnd, const unsigned int singleMask, const unsigned int doubleMask, const unsigned int initialMask)
{
	// First we calculate a base pointer for the first short value that should be updated
	newSpinImagePixelType* rowStartPointer = descriptorArray + pixelBaseIndex;
	// Next, since atomicAdd() requires an integer pointer, we force a cast to an integer pointer
	// while preserving the address of the original
	unsigned int* jobBasePixelPointer = (unsigned int*)((void*)(rowStartPointer));
	// We need an aligned pointer for atomicAdd, so we zero the final two bits of the pointer.
	// We use shifts because the actual size of uintprt_t is not known. Could be 40-bit.
	unsigned int* jobAlignedPointer = (unsigned int*)((((uintptr_t)jobBasePixelPointer) >> 2) << 2);

	int pixelCount = pixelEnd - pixelStart;

	// Zero pixel counts and unchecked pixel ranges can still exist at this point.
	// The initial loop was meant to filter them out
	// pixelEnd is not clamped, resulting in a negative overflow
	// The equals check ensures zero width ranges are filtered.
	if(pixelEnd <= pixelStart)
	{
		return;
	}

	assert((unsigned long) (jobBasePixelPointer) - (unsigned long) (jobAlignedPointer) == 0 || (unsigned long) (jobBasePixelPointer) - (unsigned long) (jobAlignedPointer) == 2);

	unsigned int currentMask = doubleMask;

	// In 1 / 2 cases, the aligned pointer will have been moved back.
	// We thus need to update the latter short in the slot, and move on to the next.
	if(jobAlignedPointer < jobBasePixelPointer)
	{
		currentMask = initialMask;
		// Needed to keep the loop going on case another pixel update needs to be done
		pixelCount++;
	// The other special scenario is a single pixel at the start. This only occurs when the base pointer equals the
	// aligned pointer and the total pixel count is 1.
	} else if(pixelCount == 1) {
		currentMask = singleMask;
	}

	unsigned int jobPointerOffset = 0;

	// We need the rounding down behaviour of division here to calculate number of "full" updates
	while (pixelCount > 0)
	{
		unsigned int* updateAddress = jobAlignedPointer + jobPointerOffset;

		atomicAdd(updateAddress, currentMask);

		pixelCount -= 2;
		jobPointerOffset++;

		currentMask = pixelCount == 1 ? singleMask : doubleMask;
	}
}

__forceinline__ __device__ unsigned lane_id()
{
	unsigned ret; 
	asm volatile ("mov.u32 %0, %laneid;" : "=r"(ret));
	return ret;
}

__device__ __inline__ QSIPrecalculatedSettings calculateRotationSettings(float3 spinImageNormal) {

// Calculating the transformation factors
	QSIPrecalculatedSettings pre_settings;

	float2 sineCosineAlpha = normalize(make_float2(spinImageNormal.x, spinImageNormal.y));

	bool is_n_a_not_zero = !((abs(spinImageNormal.x) < MAX_EQUIVALENCE_ROUNDING_ERROR) && (abs(spinImageNormal.y) < MAX_EQUIVALENCE_ROUNDING_ERROR));

	if (is_n_a_not_zero)
	{
		pre_settings.alignmentProjection_n_ax = sineCosineAlpha.x;
		pre_settings.alignmentProjection_n_ay = sineCosineAlpha.y;
	}
	else
	{
		// Leave values unchanged
		pre_settings.alignmentProjection_n_ax = 1;
		pre_settings.alignmentProjection_n_ay = 0;
	}

	float transformedNormalX = transformNormalX(pre_settings, spinImageNormal);

	float2 sineCosineBeta = normalize(make_float2(transformedNormalX, spinImageNormal.z));

	bool is_n_b_not_zero = !((abs(transformedNormalX) < MAX_EQUIVALENCE_ROUNDING_ERROR) && (abs(spinImageNormal.z) < MAX_EQUIVALENCE_ROUNDING_ERROR));

	if (is_n_b_not_zero)
	{
		pre_settings.alignmentProjection_n_bx = sineCosineBeta.x;
		pre_settings.alignmentProjection_n_bz = sineCosineBeta.y; // discrepancy between axis here is because we are using a 2D vector on 3D axis.
	}
	else
	{
		// Leave input values unchanged
		pre_settings.alignmentProjection_n_bx = 1;
		pre_settings.alignmentProjection_n_bz = 0;
	}

	return pre_settings;
}

__device__ __inline__ void rasteriseTriangle(
#if ENABLE_SHARED_MEMORY_IMAGE
		newSpinImagePixelType* sharedDescriptorArray,
#else
		array<newSpinImagePixelType> descriptors,
#endif
		float3 vertices[3], GPURasterisationSettings settings)
{
	vertices[0] = transformCoordinate(vertices[0], settings);
	vertices[1] = transformCoordinate(vertices[1], settings);
	vertices[2] = transformCoordinate(vertices[2], settings);

	float3 minVector = { 0, 0, 0 };
	float3 midVector = { 0, 0, 0 };
	float3 maxVector = { 0, 0, 0 };

	float3 deltaMinMid = { 0, 0, 0 };
	float3 deltaMidMax = { 0, 0, 0 };
	float3 deltaMinMax = { 0, 0, 0 };

	// Sort vertices by z-coordinate

	int minIndex = 0;
	int midIndex = 1;
	int maxIndex = 2;
	int _temp;

	if (vertices[minIndex].z > vertices[midIndex].z)
	{
		_temp = minIndex;
		minIndex = midIndex;
		midIndex = _temp;
	}
	if (vertices[minIndex].z > vertices[maxIndex].z)
	{
		_temp = minIndex;
		minIndex = maxIndex;
		maxIndex = _temp;
	}
	if (vertices[midIndex].z > vertices[maxIndex].z)
	{
		_temp = midIndex;
		midIndex = maxIndex;
		maxIndex = _temp;
	}

	minVector = vertices[minIndex];
	midVector = vertices[midIndex];
	maxVector = vertices[maxIndex];

	// Calculate deltas

	deltaMinMid = midVector - minVector;
	deltaMidMax = maxVector - midVector;
	deltaMinMax = maxVector - minVector;

	// Horizontal triangles are most likely not to register, and cause zero divisions, so it's easier to just get rid of them.
	if (deltaMinMax.z < MAX_EQUIVALENCE_ROUNDING_ERROR)
	{
		return;
	}

	float2 minXY = { 0, 0 };
	float2 midXY = { 0, 0 };
	float2 maxXY = { 0, 0 };

	float2 deltaMinMidXY = { 0, 0 };
	float2 deltaMidMaxXY = { 0, 0 };
	float2 deltaMinMaxXY = { 0, 0 };

	int minPixels = 0;
	int maxPixels = 0;

	// Step 6: Calculate centre line
	float centreLineFactor = deltaMinMid.z / deltaMinMax.z;
	float2 centreLineDelta = centreLineFactor * make_float2(deltaMinMax.x, deltaMinMax.y);
	float2 centreLineDirection = centreLineDelta - make_float2(deltaMinMid.x, deltaMinMid.y);
	float2 centreDirection = normalize(centreLineDirection);

	// Step 7: Rotate coordinates around origin
	// From here on out, variable names follow these conventions:
	// - X: physical relative distance to closest point on intersection line
	// - Y: Distance from origin
	minXY = alignWithPositiveX(centreDirection, make_float2(minVector.x, minVector.y));
	midXY = alignWithPositiveX(centreDirection, make_float2(midVector.x, midVector.y));
	maxXY = alignWithPositiveX(centreDirection, make_float2(maxVector.x, maxVector.y));

	deltaMinMidXY = midXY - minXY;
	deltaMidMaxXY = maxXY - midXY;
	deltaMinMaxXY = maxXY - minXY;

	// Step 8: For each row, do interpolation
	minPixels = int(floor(minVector.z /** settings.oneOverPixelSize*/));
	maxPixels = int(floor(maxVector.z /** settings.oneOverPixelSize*/));

	// Ensure we only rasterise within bounds
	minPixels = clamp(minPixels, (-spinImageWidthPixels / 2), (spinImageWidthPixels / 2) - 1);
	maxPixels = clamp(maxPixels, (-spinImageWidthPixels / 2), (spinImageWidthPixels / 2) - 1);

	int jobCount = maxPixels - minPixels;

	// Filter out job batches with no work in them
	if(jobCount == 0) {
		return;
	}

	// + 1 because we go from minPixels to <= maxPixels
	jobCount++;

	jobCount = min(minPixels + jobCount, (spinImageWidthPixels / 2)) - minPixels;

	for(int jobID = 0; jobID < jobCount; jobID++) 
	{

		int jobVertexIndexIndex;
		float jobMinVectorZ;
		float jobMidVectorZ;
		float jobDeltaMinMidZ;
		float jobDeltaMidMaxZ;
		float jobShortDeltaVectorZ;
		float jobShortVectorStartZ;
		float2 jobMinXY;
		float2 jobMidXY;
		float2 jobDeltaMinMidXY;
		float2 jobDeltaMidMaxXY;
		float2 jobShortVectorStartXY;
		float2 jobShortTransformedDelta;

		int jobMinYPixels = minPixels;
		int jobPixelY = jobMinYPixels + jobID;

		jobMinXY = minXY;
		jobMidXY = midXY;

		jobMinVectorZ = minVector.z;
		jobMidVectorZ = midVector.z;

		jobDeltaMinMidZ = deltaMinMid.z;
		jobDeltaMidMaxZ = deltaMidMax.z;

		jobDeltaMinMidXY = deltaMinMidXY;

		jobDeltaMidMaxXY = deltaMidMaxXY;

		jobVertexIndexIndex = settings.vertexIndexIndex;

		// Verified: this should be <=, because it fails for the cube test case
		if (float(jobPixelY) <= jobMidVectorZ)
		{
			// shortVectorStartXY, Bottom: minXY
			jobShortVectorStartXY = jobMinXY;
			// shortVectorStart, Bottom: minVector
			jobShortVectorStartZ = jobMinVectorZ;
			// shortDeltaVector, Bottom: deltaMinMid
			jobShortDeltaVectorZ = jobDeltaMinMidZ;
			// shortTransformedDelta, Bottom: deltaMinMidXY
			jobShortTransformedDelta = jobDeltaMinMidXY;
		}
		else
		{
			// shortVectorStartXY, Top: midXY
			jobShortVectorStartXY = jobMidXY;
			// shortVectorStart, Top: midVector
			jobShortVectorStartZ = jobMidVectorZ;
			// shortDeltaVector, Top: deltaMidMax
			jobShortDeltaVectorZ = jobDeltaMidMaxZ;
			// shortTransformedDelta, Top: deltaMidMaxXY
			jobShortTransformedDelta = jobDeltaMidMaxXY;
		}

		float jobZLevel = float(jobPixelY);
		float jobLongDistanceInTriangle = jobZLevel - jobMinVectorZ;
		float jobLongInterpolationFactor = jobLongDistanceInTriangle / deltaMinMax.z;
		float jobShortDistanceInTriangle = jobZLevel - jobShortVectorStartZ;
		float jobShortInterpolationFactor = (jobShortDeltaVectorZ == 0) ? 1.0f : jobShortDistanceInTriangle / jobShortDeltaVectorZ;
		// Set value to 1 because we want to avoid a zero division, and we define the job Z level to be at its maximum height

		int jobPixelYCoordinate = jobPixelY + (spinImageWidthPixels / 2);
		// Avoid overlap situations, only rasterise is the interpolation factors are valid
		if (jobLongDistanceInTriangle > 0 && jobShortDistanceInTriangle > 0)
		{
			// y-coordinates of both interpolated values are always equal. As such we only need to interpolate that direction once.
			// They must be equal because we have aligned the direction of the horizontal-triangle plane with the x-axis.
			float jobIntersectionY = jobMinXY.y + (jobLongInterpolationFactor * deltaMinMaxXY.y);
			// The other two x-coordinates are interpolated separately.
			float jobIntersection1X = jobShortVectorStartXY.x + (jobShortInterpolationFactor * jobShortTransformedDelta.x);
			float jobIntersection2X = jobMinXY.x + (jobLongInterpolationFactor * deltaMinMaxXY.x);

			float jobIntersection1Distance = length(make_float2(jobIntersection1X, jobIntersectionY));
			float jobIntersection2Distance = length(make_float2(jobIntersection2X, jobIntersectionY));

			// Check < 0 because we omit the case where there is exactly one point with a double intersection
			bool jobHasDoubleIntersection = (jobIntersection1X * jobIntersection2X) < 0;

			// If both values are positive or both values are negative, there is no double intersection.
			// iF the signs of the two values is different, the result will be negative or 0.
			// Having different signs implies the existence of double intersections.
			float jobDoubleIntersectionDistance = abs(jobIntersectionY);

			float jobMinDistance = jobIntersection1Distance < jobIntersection2Distance ? jobIntersection1Distance : jobIntersection2Distance;
			float jobMaxDistance = jobIntersection1Distance > jobIntersection2Distance ? jobIntersection1Distance : jobIntersection2Distance;

			unsigned int jobRowStartPixels = unsigned(floor(jobMinDistance)); // * settings.oneOverPixelSize
			unsigned int jobRowEndPixels = unsigned(floor(jobMaxDistance)); // * settings.oneOverPixelSize

			// Ensure we are only rendering within bounds
			jobRowStartPixels = min((unsigned int)spinImageWidthPixels, max(0, jobRowStartPixels));
			jobRowEndPixels = min((unsigned int)spinImageWidthPixels, jobRowEndPixels);

			size_t jobSpinImageBaseIndex = size_t(jobVertexIndexIndex) * spinImageWidthPixels * spinImageWidthPixels + jobPixelYCoordinate * spinImageWidthPixels;

			// Step 9: Fill pixels
			if (jobHasDoubleIntersection)
			{
				// since this is an absolute value, it can only be 0 or higher.
				int jobDoubleIntersectionStartPixels = int(floor(jobDoubleIntersectionDistance));// * settings.oneOverPixelSize

				// rowStartPixels must already be in bounds, and doubleIntersectionStartPixels can not be smaller than 0.
				// Hence the values in this loop are in-bounds.
#if QSI_PIXEL_DATATYPE == DATATYPE_UNSIGNED_INT || QSI_PIXEL_DATATYPE == DATATYPE_FLOAT32
				for (int jobX = jobDoubleIntersectionStartPixels; jobX < jobRowStartPixels; jobX++)
				{
					// Increment pixel by 2 because 2 intersections occurred.
#if !ENABLE_SHARED_MEMORY_IMAGE
					size_t jobPixelIndex = jobSpinImageBaseIndex + jobX;
					atomicAdd(&(descriptors.content[jobPixelIndex]), 2);
#else
					int jobPixelIndex = jobPixelYCoordinate * spinImageWidthPixels + jobX;
					atomicAdd(&(sharedDescriptorArray[jobPixelIndex]), 2);
#endif

				}
#elif QSI_PIXEL_DATATYPE == DATATYPE_UNSIGNED_SHORT
	#if !ENABLE_SHARED_MEMORY_IMAGE
				int jobBaseIndex = jobSpinImageBaseIndex + jobDoubleIntersectionStartPixels;
				newSpinImagePixelType* descriptorArrayPointer = descriptors.content;
	#else
				int jobBaseIndex = jobPixelYCoordinate * spinImageWidthPixels + jobDoubleIntersectionStartPixels;
				newSpinImagePixelType* descriptorArrayPointer = sharedDescriptorArray;
	#endif
				rasteriseRow(jobBaseIndex, descriptorArrayPointer, jobDoubleIntersectionStartPixels, jobRowStartPixels, SHORT_DOUBLE_ONE_MASK, SHORT_DOUBLE_BOTH_MASK, SHORT_DOUBLE_FIRST_MASK);
#endif
				// Now that we have already covered single intersections in the range minPixels -> doubleIntersectionEndPixels, we move the starting point for the next loop.
				// Not needed because the double intersection range is always smaller than the closest edge point
				//rowStartPixels = doubleIntersectionStartPixels + 1;
			}

#if QSI_PIXEL_DATATYPE == DATATYPE_UNSIGNED_INT || QSI_PIXEL_DATATYPE == DATATYPE_FLOAT32
			// It's imperative the condition of this loop is a < comparison
			for (int jobX = jobRowStartPixels; jobX < jobRowEndPixels; jobX++)
			{
	#if !ENABLE_SHARED_MEMORY_IMAGE
				size_t jobPixelIndex = jobSpinImageBaseIndex + jobX;
				atomicAdd(&(descriptors.content[jobPixelIndex]), 1);
	#else
				int jobPixelIndex = jobPixelYCoordinate * spinImageWidthPixels + jobX;
				atomicAdd(&(sharedDescriptorArray[jobPixelIndex]), 1);
	#endif
			}
#elif QSI_PIXEL_DATATYPE == DATATYPE_UNSIGNED_SHORT
	#if !ENABLE_SHARED_MEMORY_IMAGE
			int jobBaseIndex = jobSpinImageBaseIndex + jobRowStartPixels;
			newSpinImagePixelType* descriptorArrayPointer = descriptors.content;
	#else
			int jobBaseIndex = jobPixelYCoordinate * spinImageWidthPixels + jobRowStartPixels;
			newSpinImagePixelType* descriptorArrayPointer = sharedDescriptorArray;
	#endif
			rasteriseRow(jobBaseIndex, descriptorArrayPointer, jobRowStartPixels, jobRowEndPixels, SHORT_SINGLE_ONE_MASK, SHORT_SINGLE_BOTH_MASK, SHORT_SINGLE_FIRST_MASK);
#endif
		}
	}
}

__launch_bounds__(RASTERISATION_WARP_SIZE) __global__ void generateQuasiSpinImage(
		array<newSpinImagePixelType> descriptors,
		GPURasterisationSettings settings)
{
	// One block x-coordinate per image
	settings.vertexIndexIndex = blockIdx.x;

	// Copying over precalculated values
	settings.spinImageVertex.x = settings.mesh.vertices_x[settings.vertexIndexIndex];
	settings.spinImageVertex.y = settings.mesh.vertices_y[settings.vertexIndexIndex];
	settings.spinImageVertex.z = settings.mesh.vertices_z[settings.vertexIndexIndex];

	settings.spinImageNormal.x = settings.mesh.normals_x[settings.vertexIndexIndex];
	settings.spinImageNormal.y = settings.mesh.normals_y[settings.vertexIndexIndex];
	settings.spinImageNormal.z = settings.mesh.normals_z[settings.vertexIndexIndex];

	assert(__activemask() == 0xFFFFFFFF);

#if ENABLE_SHARED_MEMORY_IMAGE
	assert(__activemask() == 0xFFFFFFFF);

	// Creating a copy of the image in shared memory, then copying it into main memory
	__shared__ newSpinImagePixelType descriptorArrayPointer[spinImageWidthPixels * spinImageWidthPixels];

	// Initialising the values in memory to 0
	for(int i = threadIdx.x; i < spinImageWidthPixels * spinImageWidthPixels; i += RASTERISATION_WARP_SIZE)
	{
		descriptorArrayPointer[i] = 0;
	}

	__syncthreads();
#endif

	const size_t triangleCount = settings.mesh.indexCount / 3;
	for (int triangleIndex = threadIdx.x;
		 triangleIndex < triangleCount;
		 triangleIndex += RASTERISATION_WARP_SIZE)
	{
		float3 vertices[3];

		size_t triangleBaseIndex = 3 * triangleIndex;

		size_t threadTriangleIndex0 = triangleBaseIndex + 0;
		size_t threadTriangleIndex1 = triangleBaseIndex + 1;
		size_t threadTriangleIndex2 = triangleBaseIndex + 2;

		vertices[0].x = settings.mesh.vertices_x[threadTriangleIndex0];
		vertices[0].y = settings.mesh.vertices_y[threadTriangleIndex0];
		vertices[0].z = settings.mesh.vertices_z[threadTriangleIndex0];

		vertices[1].x = settings.mesh.vertices_x[threadTriangleIndex1];
		vertices[1].y = settings.mesh.vertices_y[threadTriangleIndex1];
		vertices[1].z = settings.mesh.vertices_z[threadTriangleIndex1];

		vertices[2].x = settings.mesh.vertices_x[threadTriangleIndex2];
		vertices[2].y = settings.mesh.vertices_y[threadTriangleIndex2];
		vertices[2].z = settings.mesh.vertices_z[threadTriangleIndex2];

	#if ENABLE_SHARED_MEMORY_IMAGE
		rasteriseTriangle(descriptorArrayPointer, vertices, settings);
	#else
		rasteriseTriangle(descriptors, vertices, settings);
	#endif

	}
#if ENABLE_SHARED_MEMORY_IMAGE

#if QSI_PIXEL_DATATYPE == DATATYPE_UNSIGNED_INT

	__syncthreads();
	// Image finished. Copying into main memory
	// Assumption: entire warp processes same spin image
	int jobSpinImageBaseIndex = settings.vertexIndexIndex * spinImageWidthPixels * spinImageWidthPixels;

	for (int i = threadIdx.x; i < spinImageWidthPixels * spinImageWidthPixels; i += RASTERISATION_WARP_SIZE)
	{
		atomicAdd(&descriptors.content[jobSpinImageBaseIndex + i], descriptorArrayPointer[i]);
	}
#elif QSI_PIXEL_DATATYPE == DATATYPE_UNSIGNED_SHORT
	size_t jobSpinImageBaseIndex = size_t(settings.vertexIndexIndex) * spinImageWidthPixels * spinImageWidthPixels;

	unsigned int* integerBasePointer = (unsigned int*)((void*)(descriptors.content + jobSpinImageBaseIndex));
	unsigned int* sharedImageIntPointer = (unsigned int*)((void*)(descriptorArrayPointer));

	// Divide update count by 2 because we update two pixels at a time
	for (int i = threadIdx.x; i < (spinImageWidthPixels * spinImageWidthPixels) / 2; i += RASTERISATION_WARP_SIZE)
	{
		atomicAdd(integerBasePointer + i, *(sharedImageIntPointer + i));
	}
#endif
#endif

}

array<newSpinImagePixelType> generateQuasiSpinImages(DeviceMesh device_mesh, cudaDeviceProp device_information,
													 float spinImageWidth)
{
	size_t descriptorBufferLength = device_mesh.vertexCount * spinImageWidthPixels * spinImageWidthPixels;
	size_t descriptorBufferSize = sizeof(newSpinImagePixelType) * descriptorBufferLength;

	array<newSpinImagePixelType> device_descriptors;
	checkCudaErrors(cudaMalloc(&device_descriptors.content, descriptorBufferSize));

	size_t imageCount = device_mesh.vertexCount;
	device_descriptors.length = imageCount;

	CudaLaunchDimensions valueSetSettings = calculateCudaLaunchDimensions(descriptorBufferLength, device_information);
	setValue<newSpinImagePixelType><< <valueSetSettings.blocksPerGrid, valueSetSettings.threadsPerBlock >> > (device_descriptors.content, descriptorBufferLength, 0);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	auto start = std::chrono::steady_clock::now();

	GPURasterisationSettings generalSettings;
	generalSettings.mesh = device_mesh;

	generateQuasiSpinImage <<<imageCount, RASTERISATION_WARP_SIZE>>> (device_descriptors, generalSettings);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	std::chrono::milliseconds duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
	std::cout << "Execution time:" << duration.count() << std::endl;

    array<newSpinImagePixelType> host_descriptors;
	host_descriptors.content = new newSpinImagePixelType[imageCount * spinImageWidthPixels * spinImageWidthPixels];
	host_descriptors.length = imageCount;

	checkCudaErrors(cudaMemcpy(host_descriptors.content, device_descriptors.content, descriptorBufferSize, cudaMemcpyDeviceToHost));

	cudaFree(device_descriptors.content);

	return host_descriptors;
}

