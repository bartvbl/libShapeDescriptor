#include "quasiSpinImageGenerator.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <spinImage/gpu/types/Mesh.h>
#include <spinImage/gpu/types/CudaLaunchDimensions.h>
#include <spinImage/utilities/setValue.cuh>
#include <spinImage/libraryBuildSettings.h>

#include "nvidia/helper_math.h"
#include "nvidia/helper_cuda.h"

#include <cassert>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <sstream>
#include <spinImage/gpu/types/DeviceOrientedPoint.h>

#if RICI_PIXEL_DATATYPE == DATATYPE_UNSIGNED_SHORT
const int SHORT_SINGLE_BOTH_MASK = 0x00010001;
const int SHORT_SINGLE_ONE_MASK = 0x00000001;
const int SHORT_SINGLE_FIRST_MASK = 0x00010000;

const int SHORT_DOUBLE_BOTH_MASK = 0x00020002;
const int SHORT_DOUBLE_ONE_MASK = 0x00000002;
const int SHORT_DOUBLE_FIRST_MASK = 0x00020000;
#endif

#define spinOriginCount gridDim.x
#define renderedSpinImageIndex blockIdx.x

const int RASTERISATION_WARP_SIZE = 1024;

struct RICIMesh {
    float* geometryBasePointer;
    float* spinOriginsBasePointer;

    size_t vertexCount;
};

__device__ __inline__ float3 transformCoordinate(const float3 &vertex, const float3 &spinImageVertex, const float3 &spinImageNormal)
{
    const float2 sineCosineAlpha = normalize(make_float2(spinImageNormal.x, spinImageNormal.y));

    const bool is_n_a_not_zero = !((abs(spinImageNormal.x) < MAX_EQUIVALENCE_ROUNDING_ERROR) && (abs(spinImageNormal.y) < MAX_EQUIVALENCE_ROUNDING_ERROR));

    const float alignmentProjection_n_ax = is_n_a_not_zero ? sineCosineAlpha.x : 1;
    const float alignmentProjection_n_ay = is_n_a_not_zero ? sineCosineAlpha.y : 0;

	float3 transformedCoordinate = vertex - spinImageVertex;

	const float initialTransformedX = transformedCoordinate.x;
	transformedCoordinate.x = alignmentProjection_n_ax * transformedCoordinate.x + alignmentProjection_n_ay * transformedCoordinate.y;
	transformedCoordinate.y = -alignmentProjection_n_ay * initialTransformedX + alignmentProjection_n_ax * transformedCoordinate.y;

    const float transformedNormalX = alignmentProjection_n_ax * spinImageNormal.x + alignmentProjection_n_ay * spinImageNormal.y;

    const float2 sineCosineBeta = normalize(make_float2(transformedNormalX, spinImageNormal.z));

    const bool is_n_b_not_zero = !((abs(transformedNormalX) < MAX_EQUIVALENCE_ROUNDING_ERROR) && (abs(spinImageNormal.z) < MAX_EQUIVALENCE_ROUNDING_ERROR));

    const float alignmentProjection_n_bx = is_n_b_not_zero ? sineCosineBeta.x : 1;
    const float alignmentProjection_n_bz = is_n_b_not_zero ? sineCosineBeta.y : 0; // discrepancy between axis here is because we are using a 2D vector on 3D axis.

	// Order matters here
	const float initialTransformedX_2 = transformedCoordinate.x;
	transformedCoordinate.x = alignmentProjection_n_bz * transformedCoordinate.x - alignmentProjection_n_bx * transformedCoordinate.z;
	transformedCoordinate.z = alignmentProjection_n_bx * initialTransformedX_2 + alignmentProjection_n_bz * transformedCoordinate.z;

	return transformedCoordinate;
}

__device__ __inline__ float2 alignWithPositiveX(const float2 &midLineDirection, const float2 &vertex)
{
	float2 transformed;
	transformed.x = midLineDirection.x * vertex.x + midLineDirection.y * vertex.y;
	transformed.y = -midLineDirection.y * vertex.x + midLineDirection.x * vertex.y;
	return transformed;
}

__host__ __device__ __inline__ size_t roundSizeToNearestCacheLine(size_t sizeInBytes) {
    return (sizeInBytes + 127u) & ~((size_t) 127);
}

#if RICI_PIXEL_DATATYPE == DATATYPE_UNSIGNED_SHORT
__device__ __inline__ void rasteriseRow(int pixelBaseIndex, quasiSpinImagePixelType* descriptorArray, unsigned int pixelStart, unsigned int pixelEnd, const unsigned int singleMask, const unsigned int doubleMask, const unsigned int initialMask)
{
	// First we calculate a base pointer for the first short value that should be updated
	quasiSpinImagePixelType* rowStartPointer = descriptorArray + pixelBaseIndex;
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
#endif

__device__ __inline__ void rasteriseTriangle(
		quasiSpinImagePixelType* descriptors,
		float3 vertices[3],
		const float3 &spinImageVertex,
		const float3 &spinImageNormal)
{
	vertices[0] = transformCoordinate(vertices[0], spinImageVertex, spinImageNormal);
	vertices[1] = transformCoordinate(vertices[1], spinImageVertex, spinImageNormal);
	vertices[2] = transformCoordinate(vertices[2], spinImageVertex, spinImageNormal);

	// Sort vertices by z-coordinate

	char minIndex = 0;
    char midIndex = 1;
    char maxIndex = 2;
    char _temp;

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

    const float3 minVector = vertices[minIndex];
	const float3 midVector = vertices[midIndex];
	const float3 maxVector = vertices[maxIndex];

	// Calculate deltas

    const float3 deltaMinMid = midVector - minVector;
    const float3 deltaMidMax = maxVector - midVector;
    const float3 deltaMinMax = maxVector - minVector;

	// Horizontal triangles are most likely not to register, and cause zero divisions, so it's easier to just get rid of them.
	if (deltaMinMax.z < MAX_EQUIVALENCE_ROUNDING_ERROR)
	{
		return;
	}

	// Step 6: Calculate centre line
	const float centreLineFactor = deltaMinMid.z / deltaMinMax.z;
    const float2 centreLineDelta = centreLineFactor * make_float2(deltaMinMax.x, deltaMinMax.y);
    const float2 centreLineDirection = centreLineDelta - make_float2(deltaMinMid.x, deltaMinMid.y);
    const float2 centreDirection = normalize(centreLineDirection);

	// Step 7: Rotate coordinates around origin
	// From here on out, variable names follow these conventions:
	// - X: physical relative distance to closest point on intersection line
	// - Y: Distance from origin
	const float2 minXY = alignWithPositiveX(centreDirection, make_float2(minVector.x, minVector.y));
    const float2 midXY = alignWithPositiveX(centreDirection, make_float2(midVector.x, midVector.y));
    const float2 maxXY = alignWithPositiveX(centreDirection, make_float2(maxVector.x, maxVector.y));

    const float2 deltaMinMidXY = midXY - minXY;
    const float2 deltaMidMaxXY = maxXY - midXY;
    const float2 deltaMinMaxXY = maxXY - minXY;

	// Step 8: For each row, do interpolation
	// And ensure we only rasterise within bounds
	const short minPixels = clamp(short(floor(minVector.z)), (-spinImageWidthPixels / 2), (spinImageWidthPixels / 2) - 1);
	const short maxPixels = clamp(short(floor(maxVector.z)), (-spinImageWidthPixels / 2), (spinImageWidthPixels / 2) - 1);

	int pixelRowCount = maxPixels - minPixels;

	// Filter out job batches with no work in them
	if(pixelRowCount == 0) {
		return;
	}

	// + 1 because we go from minPixels to <= maxPixels
	pixelRowCount++;

	pixelRowCount = min(minPixels + pixelRowCount, (spinImageWidthPixels / 2)) - minPixels;

	for(short pixelRowID = 0; pixelRowID < pixelRowCount; pixelRowID++)
	{
		const short pixelY = minPixels + pixelRowID;

		// Verified: this should be <=, because it fails for the cube tests case
		const bool isBottomSection = float(pixelY) <= midVector.z;

		// Technically I can rewrite this into two separate loops
		// However, that would increase the thread divergence
		// I believe this is the best option
		const float shortDeltaVectorZ = isBottomSection ? deltaMinMid.z : deltaMidMax.z;
		const float shortVectorStartZ = isBottomSection ? minVector.z : midVector.z;
		const float2 shortVectorStartXY = isBottomSection ? minXY : midXY;
		const float2 shortTransformedDelta = isBottomSection ? deltaMinMidXY : deltaMidMaxXY;

		const float zLevel = float(pixelY);
		const float longDistanceInTriangle = zLevel - minVector.z;
        const float longInterpolationFactor = longDistanceInTriangle / deltaMinMax.z;
        const float shortDistanceInTriangle = zLevel - shortVectorStartZ;
        const float shortInterpolationFactor = (shortDeltaVectorZ == 0) ? 1.0f : shortDistanceInTriangle / shortDeltaVectorZ;
		// Set value to 1 because we want to avoid a zero division, and we define the job Z level to be at its maximum height

        const unsigned short pixelYCoordinate = (unsigned short)(pixelY + (spinImageWidthPixels / 2));
		// Avoid overlap situations, only rasterise is the interpolation factors are valid
		if (longDistanceInTriangle > 0 && shortDistanceInTriangle > 0)
		{
			// y-coordinates of both interpolated values are always equal. As such we only need to interpolate that direction once.
			// They must be equal because we have aligned the direction of the horizontal-triangle plane with the x-axis.
			const float intersectionY = minXY.y + (longInterpolationFactor * deltaMinMaxXY.y);
			// The other two x-coordinates are interpolated separately.
            const float intersection1X = shortVectorStartXY.x + (shortInterpolationFactor * shortTransformedDelta.x);
            const float intersection2X = minXY.x + (longInterpolationFactor * deltaMinMaxXY.x);

            const float intersection1Distance = length(make_float2(intersection1X, intersectionY));
            const float intersection2Distance = length(make_float2(intersection2X, intersectionY));

			// Check < 0 because we omit the case where there is exactly one point with a double intersection
            const bool hasDoubleIntersection = (intersection1X * intersection2X) < 0;

			// If both values are positive or both values are negative, there is no double intersection.
			// iF the signs of the two values is different, the result will be negative or 0.
			// Having different signs implies the existence of double intersections.
            const float doubleIntersectionDistance = abs(intersectionY);

            const float minDistance = intersection1Distance < intersection2Distance ? intersection1Distance : intersection2Distance;
            const float maxDistance = intersection1Distance > intersection2Distance ? intersection1Distance : intersection2Distance;

            unsigned short rowStartPixels = (unsigned short) (floor(minDistance));
            unsigned short rowEndPixels = (unsigned short) (floor(maxDistance));

			// Ensure we are only rendering within bounds
			rowStartPixels = min((unsigned int)spinImageWidthPixels, max(0, rowStartPixels));
			rowEndPixels = min((unsigned int)spinImageWidthPixels, rowEndPixels);

#if !ENABLE_SHARED_MEMORY_IMAGE
			const size_t jobSpinImageBaseIndex = size_t(renderedSpinImageIndex) * spinImageWidthPixels * spinImageWidthPixels + pixelYCoordinate * spinImageWidthPixels;
#else
			const size_t jobSpinImageBaseIndex = pixelYCoordinate * ((unsigned short) spinImageWidthPixels);
#endif

			// Step 9: Fill pixels
			if (hasDoubleIntersection)
			{
				// since this is an absolute value, it can only be 0 or higher.
				const int jobDoubleIntersectionStartPixels = int(floor(doubleIntersectionDistance));

				// rowStartPixels must already be in bounds, and doubleIntersectionStartPixels can not be smaller than 0.
				// Hence the values in this loop are in-bounds.
#if RICI_PIXEL_DATATYPE == DATATYPE_UNSIGNED_INT || RICI_PIXEL_DATATYPE == DATATYPE_FLOAT32
				for (int jobX = jobDoubleIntersectionStartPixels; jobX < rowStartPixels; jobX++)
				{
					// Increment pixel by 2 because 2 intersections occurred.
					atomicAdd(&(descriptors[jobSpinImageBaseIndex + jobX]), 2);
				}
#elif RICI_PIXEL_DATATYPE == DATATYPE_UNSIGNED_SHORT
	#if !ENABLE_SHARED_MEMORY_IMAGE
				int jobBaseIndex = jobSpinImageBaseIndex + jobDoubleIntersectionStartPixels;
				quasiSpinImagePixelType* descriptorArrayPointer = descriptors.content;
	#else
				int jobBaseIndex = pixelYCoordinate * spinImageWidthPixels + jobDoubleIntersectionStartPixels;
				quasiSpinImagePixelType* descriptorArrayPointer = descriptors;
	#endif
				rasteriseRow(jobBaseIndex, descriptorArrayPointer, jobDoubleIntersectionStartPixels, rowStartPixels, SHORT_DOUBLE_ONE_MASK, SHORT_DOUBLE_BOTH_MASK, SHORT_DOUBLE_FIRST_MASK);
#endif
			}

#if RICI_PIXEL_DATATYPE == DATATYPE_UNSIGNED_INT || RICI_PIXEL_DATATYPE == DATATYPE_FLOAT32
			// It's imperative the condition of this loop is a < comparison
			for (int jobX = rowStartPixels; jobX < rowEndPixels; jobX++)
			{
				atomicAdd(&(descriptors[jobSpinImageBaseIndex + jobX]), 1);
			}
#elif RICI_PIXEL_DATATYPE == DATATYPE_UNSIGNED_SHORT
	#if !ENABLE_SHARED_MEMORY_IMAGE
			int jobBaseIndex = jobSpinImageBaseIndex + rowStartPixels;
			quasiSpinImagePixelType* descriptorArrayPointer = descriptors.content;
	#else
			int jobBaseIndex = pixelYCoordinate * spinImageWidthPixels + rowStartPixels;
			quasiSpinImagePixelType* descriptorArrayPointer = descriptors;
	#endif
			rasteriseRow(jobBaseIndex, descriptorArrayPointer, rowStartPixels, rowEndPixels, SHORT_SINGLE_ONE_MASK, SHORT_SINGLE_BOTH_MASK, SHORT_SINGLE_FIRST_MASK);
#endif
		}
	}
}

__launch_bounds__(RASTERISATION_WARP_SIZE, 2) __global__ void generateQuasiSpinImage(
		quasiSpinImagePixelType* descriptors,
        RICIMesh mesh)
{
	// Copying over precalculated values
	float3 spinImageVertex;

	const size_t spinComponentBlockSize = roundSizeToNearestCacheLine(spinOriginCount);

	spinImageVertex.x = mesh.spinOriginsBasePointer[0 * spinComponentBlockSize + renderedSpinImageIndex];
    spinImageVertex.y = mesh.spinOriginsBasePointer[1 * spinComponentBlockSize + renderedSpinImageIndex];
	spinImageVertex.z = mesh.spinOriginsBasePointer[2 * spinComponentBlockSize + renderedSpinImageIndex];

	float3 spinImageNormal;
	spinImageNormal.x = mesh.spinOriginsBasePointer[3 * spinComponentBlockSize + renderedSpinImageIndex];
	spinImageNormal.y = mesh.spinOriginsBasePointer[4 * spinComponentBlockSize + renderedSpinImageIndex];
	spinImageNormal.z = mesh.spinOriginsBasePointer[5 * spinComponentBlockSize + renderedSpinImageIndex];

	assert(__activemask() == 0xFFFFFFFF);

#if ENABLE_SHARED_MEMORY_IMAGE
	// Creating a copy of the image in shared memory, then copying it into main memory
	__shared__ quasiSpinImagePixelType descriptorArrayPointer[spinImageWidthPixels * spinImageWidthPixels];

	// Initialising the values in memory to 0
	for(int i = threadIdx.x; i < spinImageWidthPixels * spinImageWidthPixels; i += RASTERISATION_WARP_SIZE)
	{
		descriptorArrayPointer[i] = 0;
	}

	__syncthreads();
#endif

	const size_t triangleCount = mesh.vertexCount / 3;
	for (int triangleIndex = threadIdx.x;
		 triangleIndex < triangleCount;
		 triangleIndex += RASTERISATION_WARP_SIZE)
	{
		float3 vertices[3];

        const size_t vertexComponentBlockSize = roundSizeToNearestCacheLine(triangleCount);

		vertices[0].x = mesh.geometryBasePointer[0 * vertexComponentBlockSize + triangleIndex];
		vertices[0].y = mesh.geometryBasePointer[1 * vertexComponentBlockSize + triangleIndex];
		vertices[0].z = mesh.geometryBasePointer[2 * vertexComponentBlockSize + triangleIndex];

		vertices[1].x = mesh.geometryBasePointer[3 * vertexComponentBlockSize + triangleIndex];
		vertices[1].y = mesh.geometryBasePointer[4 * vertexComponentBlockSize + triangleIndex];
		vertices[1].z = mesh.geometryBasePointer[5 * vertexComponentBlockSize + triangleIndex];

		vertices[2].x = mesh.geometryBasePointer[6 * vertexComponentBlockSize + triangleIndex];
		vertices[2].y = mesh.geometryBasePointer[7 * vertexComponentBlockSize + triangleIndex];
		vertices[2].z = mesh.geometryBasePointer[8 * vertexComponentBlockSize + triangleIndex];

	#if ENABLE_SHARED_MEMORY_IMAGE
		rasteriseTriangle(descriptorArrayPointer, vertices, spinImageVertex, spinImageNormal);
	#else
		rasteriseTriangle(descriptors, vertices, settings);
	#endif

	}
#if ENABLE_SHARED_MEMORY_IMAGE

#if RICI_PIXEL_DATATYPE == DATATYPE_UNSIGNED_INT

	__syncthreads();
	// Image finished. Copying into main memory
	// Assumption: entire warp processes same spin image
	const size_t jobSpinImageBaseIndex = renderedSpinImageIndex * spinImageWidthPixels * spinImageWidthPixels;

	for (int i = threadIdx.x; i < spinImageWidthPixels * spinImageWidthPixels; i += RASTERISATION_WARP_SIZE)
	{
		atomicAdd(&descriptors[jobSpinImageBaseIndex + i], descriptorArrayPointer[i]);
	}
#elif RICI_PIXEL_DATATYPE == DATATYPE_UNSIGNED_SHORT
	size_t jobSpinImageBaseIndex = size_t(renderedSpinImageIndex) * spinImageWidthPixels * spinImageWidthPixels;

	unsigned int* integerBasePointer = (unsigned int*)((void*)(descriptors + jobSpinImageBaseIndex));
	unsigned int* sharedImageIntPointer = (unsigned int*)((void*)(descriptorArrayPointer));

	// Divide update count by 2 because we update two pixels at a time
	for (int i = threadIdx.x; i < (spinImageWidthPixels * spinImageWidthPixels) / 2; i += RASTERISATION_WARP_SIZE)
	{
		atomicAdd(integerBasePointer + i, *(sharedImageIntPointer + i));
	}
#endif
#endif
}

__global__ void scaleMesh(SpinImage::gpu::Mesh mesh, float scaleFactor) {
    size_t vertexIndex = blockDim.x * blockIdx.x + threadIdx.x;

    if(vertexIndex >= mesh.vertexCount) {
        return;
    }

    mesh.vertices_x[vertexIndex] *= scaleFactor;
    mesh.vertices_y[vertexIndex] *= scaleFactor;
    mesh.vertices_z[vertexIndex] *= scaleFactor;
}

__global__ void scaleSpinOrigins(SpinImage::gpu::DeviceOrientedPoint* origins, size_t imageCount, float scaleFactor) {
    size_t vertexIndex = blockDim.x * blockIdx.x + threadIdx.x;

    if(vertexIndex >= imageCount) {
        return;
    }

    origins[vertexIndex].vertex.x *= scaleFactor;
    origins[vertexIndex].vertex.y *= scaleFactor;
    origins[vertexIndex].vertex.z *= scaleFactor;
}

__global__ void redistributeMesh(SpinImage::gpu::Mesh mesh, RICIMesh riciMesh) {
    size_t triangleIndex = blockIdx.x;
    size_t triangleBaseIndex = 3 * triangleIndex;

    // Round up the triangle count to the nearest
    size_t triangleCount = mesh.vertexCount / 3;
    size_t geometryBlockSize = roundSizeToNearestCacheLine(triangleCount);

    riciMesh.geometryBasePointer[0 * geometryBlockSize + triangleIndex] = mesh.vertices_x[triangleBaseIndex + 0];
    riciMesh.geometryBasePointer[1 * geometryBlockSize + triangleIndex] = mesh.vertices_y[triangleBaseIndex + 0];
    riciMesh.geometryBasePointer[2 * geometryBlockSize + triangleIndex] = mesh.vertices_z[triangleBaseIndex + 0];
    riciMesh.geometryBasePointer[3 * geometryBlockSize + triangleIndex] = mesh.vertices_x[triangleBaseIndex + 1];
    riciMesh.geometryBasePointer[4 * geometryBlockSize + triangleIndex] = mesh.vertices_y[triangleBaseIndex + 1];
    riciMesh.geometryBasePointer[5 * geometryBlockSize + triangleIndex] = mesh.vertices_z[triangleBaseIndex + 1];
    riciMesh.geometryBasePointer[6 * geometryBlockSize + triangleIndex] = mesh.vertices_x[triangleBaseIndex + 2];
    riciMesh.geometryBasePointer[7 * geometryBlockSize + triangleIndex] = mesh.vertices_y[triangleBaseIndex + 2];
    riciMesh.geometryBasePointer[8 * geometryBlockSize + triangleIndex] = mesh.vertices_z[triangleBaseIndex + 2];
}

__global__ void redistributeSpinOrigins(SpinImage::gpu::DeviceOrientedPoint* spinOrigins, size_t imageCount, RICIMesh riciMesh) {
    assert(imageCount == gridDim.x);
    size_t imageIndex = blockIdx.x;

    size_t spinOriginsBlockSize = roundSizeToNearestCacheLine(imageCount);

    SpinImage::gpu::DeviceOrientedPoint spinOrigin = spinOrigins[imageIndex];

    riciMesh.spinOriginsBasePointer[0 * spinOriginsBlockSize + imageIndex] = spinOrigin.vertex.x;
    riciMesh.spinOriginsBasePointer[1 * spinOriginsBlockSize + imageIndex] = spinOrigin.vertex.y;
    riciMesh.spinOriginsBasePointer[2 * spinOriginsBlockSize + imageIndex] = spinOrigin.vertex.z;

    riciMesh.spinOriginsBasePointer[3 * spinOriginsBlockSize + imageIndex] = spinOrigin.normal.x;
    riciMesh.spinOriginsBasePointer[4 * spinOriginsBlockSize + imageIndex] = spinOrigin.normal.y;
    riciMesh.spinOriginsBasePointer[5 * spinOriginsBlockSize + imageIndex] = spinOrigin.normal.z;
}

SpinImage::array<quasiSpinImagePixelType> SpinImage::gpu::generateQuasiSpinImages(
        Mesh device_mesh,
        array<DeviceOrientedPoint> device_RICIOrigins,
        float spinImageWidth,
        SpinImage::debug::RICIRunInfo* runinfo)
{
    auto totalExecutionTimeStart = std::chrono::steady_clock::now();

    size_t imageCount = device_RICIOrigins.length;
    size_t meshVertexCount = device_mesh.vertexCount;
    size_t triangleCount = meshVertexCount / (size_t) 3;

    // -- Mesh Scaling --
    // The mesh is scaled to where 1 distance unit is one pixel.
    // This saves a lot of divisions when rendering the images.

	auto meshScaleTimeStart = std::chrono::steady_clock::now();

	    float scaleFactor = float(spinImageWidthPixels)/spinImageWidth;

	    Mesh device_editableMeshCopy = duplicateMesh(device_mesh);
	    scaleMesh<<<(meshVertexCount / 128) + 1, 128>>>(device_editableMeshCopy, scaleFactor);
        checkCudaErrors(cudaDeviceSynchronize());

        DeviceOrientedPoint* device_editableSpinOriginsCopy;
        checkCudaErrors(cudaMalloc(&device_editableSpinOriginsCopy, imageCount * sizeof(DeviceOrientedPoint)));
        checkCudaErrors(cudaMemcpy(device_editableSpinOriginsCopy, device_RICIOrigins.content, imageCount * sizeof(DeviceOrientedPoint), cudaMemcpyDeviceToDevice));
        scaleSpinOrigins<<<(imageCount / 128) + 1, 128>>>(device_editableSpinOriginsCopy, imageCount, scaleFactor);
        checkCudaErrors(cudaDeviceSynchronize());

    std::chrono::milliseconds meshScaleDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - meshScaleTimeStart);

    // -- Array Restructuring --
    // In order to optimise memory access patterns, we rearrange the memory layout of the input data

    RICIMesh riciMesh;
    riciMesh.vertexCount = meshVertexCount;

    checkCudaErrors(cudaMalloc(&riciMesh.spinOriginsBasePointer,
            roundSizeToNearestCacheLine(imageCount) * 2 * sizeof(float3)));
    checkCudaErrors(cudaMalloc(&riciMesh.geometryBasePointer,
            roundSizeToNearestCacheLine(meshVertexCount) * 3 * sizeof(float3)));

    auto redistributeTimeStart = std::chrono::steady_clock::now();

        redistributeMesh<<<triangleCount, 1>>>(device_editableMeshCopy, riciMesh);
        redistributeSpinOrigins<<<imageCount, 1>>>(device_editableSpinOriginsCopy, imageCount, riciMesh);
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaGetLastError());

    std::chrono::milliseconds redistributeDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - redistributeTimeStart);

    // -- Descriptor Array Allocation and Initialisation --

    size_t descriptorBufferLength = imageCount * spinImageWidthPixels * spinImageWidthPixels;
    size_t descriptorBufferSize = sizeof(quasiSpinImagePixelType) * descriptorBufferLength;


    array<quasiSpinImagePixelType> device_descriptors;
	checkCudaErrors(cudaMalloc(&device_descriptors.content, descriptorBufferSize));
	device_descriptors.length = imageCount;

	setValue<quasiSpinImagePixelType><<<(descriptorBufferLength / 64) + 1, 64>>> (device_descriptors.content, descriptorBufferLength, 0);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    // -- Descriptor Generation --

	auto generationStart = std::chrono::steady_clock::now();

	    // Warning: kernel assumes the grid dimensions are equivalent to imageCount.
	    generateQuasiSpinImage <<<imageCount, RASTERISATION_WARP_SIZE>>> (device_descriptors.content, riciMesh);
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaGetLastError());

	std::chrono::milliseconds generationDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - generationStart);

	// -- Cleanup --

    freeMesh(device_editableMeshCopy);
	cudaFree(riciMesh.spinOriginsBasePointer);
	cudaFree(riciMesh.geometryBasePointer);
	cudaFree(device_editableSpinOriginsCopy);

    std::chrono::milliseconds totalExecutionDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - totalExecutionTimeStart);

    if(runinfo != nullptr) {
        runinfo->totalExecutionTimeSeconds = double(totalExecutionDuration.count()) / 1000.0;
	    runinfo->generationTimeSeconds = double(generationDuration.count()) / 1000.0;
	    runinfo->meshScaleTimeSeconds = double(meshScaleDuration.count()) / 1000.0;
	    runinfo->redistributionTimeSeconds = double(redistributeDuration.count()) / 1000.0;
	}

    return device_descriptors;
}