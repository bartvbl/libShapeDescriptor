#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
#include "cuda_runtime.h"
#include "helper_math.h"
#include "helper_cuda.h"
#include "device_launch_parameters.h"
#endif

#include <shapeDescriptor/shapeDescriptor.h>

#include <shapeDescriptor/libraryBuildSettings.h>

#include <cassert>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <sstream>

#ifndef ENABLE_SHARED_MEMORY_IMAGE
#define ENABLE_SHARED_MEMORY_IMAGE true
#endif

// Code is made for unsigned integers. Shorts would require additional logic.
static_assert(sizeof(radialIntersectionCountImagePixelType) == 4, "Code is made for unsigned integers");

#define spinOriginCount gridDim.x
#define renderedSpinImageIndex blockIdx.x

const int RASTERISATION_WARP_SIZE = 1024;

struct QUICCIMesh {
    float* geometryBasePointer;
    float* spinOriginsBasePointer;

    size_t vertexCount;
};

#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
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

__device__ __inline__ void rasteriseTriangle(
        radialIntersectionCountImagePixelType* descriptors,
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
    const int minPixels = int(floor(minVector.z));
    const int maxPixels = int(floor(maxVector.z));

    const int halfHeight = spinImageWidthPixels / 2;

    // Filter out job batches with no work in them
    if((minPixels < -halfHeight && maxPixels < -halfHeight) ||
       (minPixels >= halfHeight && maxPixels >= halfHeight)) {
        return;
    }

    const int startRowIndex = max(-halfHeight, minPixels);
    const int endRowIndex = min(halfHeight - 1, maxPixels);

    for(int pixelY = startRowIndex; pixelY <= endRowIndex; pixelY++)
    {
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

            const size_t jobSpinImageBaseIndex = pixelYCoordinate * ((unsigned short) spinImageWidthPixels);

            // Step 9: Fill pixels
            if (hasDoubleIntersection)
            {
                // since this is an absolute value, it can only be 0 or higher.
                const int jobDoubleIntersectionStartPixels = int(floor(doubleIntersectionDistance));

                // rowStartPixels must already be in bounds, and doubleIntersectionStartPixels can not be smaller than 0.
                // Hence the values in this loop are in-bounds.
                for (int jobX = jobDoubleIntersectionStartPixels; jobX < rowStartPixels; jobX++)
                {
                    // Increment pixel by 2 because 2 intersections occurred.
                    atomicAdd(&(descriptors[jobSpinImageBaseIndex + jobX]), 2);
                }
            }

            // It's imperative the condition of this loop is a < comparison
            for (int jobX = rowStartPixels; jobX < rowEndPixels; jobX++)
            {
                atomicAdd(&(descriptors[jobSpinImageBaseIndex + jobX]), 1);
            }
        }
    }
}


template<unsigned int pixelDifferenceThreshold>
__device__ void writeQUICCImage(
        ShapeDescriptor::QUICCIDescriptor* descriptorArray,
        radialIntersectionCountImagePixelType* RICIDescriptor) {

    const int laneIndex = threadIdx.x % 32;
    static_assert(spinImageWidthPixels % 32 == 0, "Implementation assumes images are multiples of warps wide");

    for(int row = 0; row < spinImageWidthPixels; row++) {
        radialIntersectionCountImagePixelType previousWarpLastNeedlePixelValue = 0;

        for (int pixel = laneIndex; pixel < spinImageWidthPixels; pixel += warpSize) {
            radialIntersectionCountImagePixelType currentNeedlePixelValue =
                    RICIDescriptor[row * spinImageWidthPixels + pixel];

            int targetThread;
            if (laneIndex > 0) {
                targetThread = laneIndex - 1;
            }
            else if (pixel > 0) {
                targetThread = 31;
            } else {
                targetThread = 0;
            }

            radialIntersectionCountImagePixelType threadNeedleValue = 0;

            if (laneIndex == 31) {
                threadNeedleValue = previousWarpLastNeedlePixelValue;
            } else {
                threadNeedleValue = currentNeedlePixelValue;
            }

            radialIntersectionCountImagePixelType previousNeedlePixelValue = __shfl_sync(0xFFFFFFFF, threadNeedleValue, targetThread);

            int imageDelta = int(currentNeedlePixelValue) - int(previousNeedlePixelValue);

            bool didIntersectionCountsChange = (imageDelta * imageDelta) >= (pixelDifferenceThreshold * pixelDifferenceThreshold);

            unsigned int changeOccurredCombined = __brev(__ballot_sync(0xFFFFFFFF, didIntersectionCountsChange));

            if(laneIndex == 0) {
                size_t chunkIndex = (row * (spinImageWidthPixels / 32)) + (pixel / 32);
                descriptorArray[blockIdx.x].contents[chunkIndex] = changeOccurredCombined;
            }

            // This only matters for thread 31, so no need to broadcast it using a shuffle instruction
            previousWarpLastNeedlePixelValue = currentNeedlePixelValue;
        }

    }
}

template<unsigned int pixelDifferenceThreshold>
__launch_bounds__(RASTERISATION_WARP_SIZE, 2) __global__ void generateQuickIntersectionCountChangeImage(
        ShapeDescriptor::QUICCIDescriptor* descriptors,
        QUICCIMesh mesh)
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
    // Creating a copy of the image in shared memory, then copying it into main memory
	__shared__ radialIntersectionCountImagePixelType descriptorArrayPointer[spinImageWidthPixels * spinImageWidthPixels];

	// Initialising the values in memory to 0
	for(int i = threadIdx.x; i < spinImageWidthPixels * spinImageWidthPixels; i += RASTERISATION_WARP_SIZE)
	{
		descriptorArrayPointer[i] = 0;
	}

	__syncthreads();

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

        rasteriseTriangle(descriptorArrayPointer, vertices, spinImageVertex, spinImageNormal);
    }

	__syncthreads();
	// Image finished. Copying into main memory
	writeQUICCImage<pixelDifferenceThreshold>(descriptors, descriptorArrayPointer);
}

__global__ void scaleQUICCIMesh(ShapeDescriptor::gpu::Mesh mesh, float scaleFactor) {
    size_t vertexIndex = blockDim.x * blockIdx.x + threadIdx.x;

    if(vertexIndex >= mesh.vertexCount) {
        return;
    }

    mesh.vertices_x[vertexIndex] *= scaleFactor;
    mesh.vertices_y[vertexIndex] *= scaleFactor;
    mesh.vertices_z[vertexIndex] *= scaleFactor;
}

__global__ void scaleQUICCISpinOrigins(ShapeDescriptor::OrientedPoint* origins, size_t imageCount, float scaleFactor) {
    size_t vertexIndex = blockDim.x * blockIdx.x + threadIdx.x;

    if(vertexIndex >= imageCount) {
        return;
    }

    origins[vertexIndex].vertex.x *= scaleFactor;
    origins[vertexIndex].vertex.y *= scaleFactor;
    origins[vertexIndex].vertex.z *= scaleFactor;
}

__global__ void redistributeMesh(ShapeDescriptor::gpu::Mesh mesh, QUICCIMesh quicciMesh) {
    size_t triangleIndex = blockIdx.x;
    size_t triangleBaseIndex = 3 * triangleIndex;

    // Round up the triangle count to the nearest
    size_t triangleCount = mesh.vertexCount / 3;
    size_t geometryBlockSize = roundSizeToNearestCacheLine(triangleCount);

    quicciMesh.geometryBasePointer[0 * geometryBlockSize + triangleIndex] = mesh.vertices_x[triangleBaseIndex + 0];
    quicciMesh.geometryBasePointer[1 * geometryBlockSize + triangleIndex] = mesh.vertices_y[triangleBaseIndex + 0];
    quicciMesh.geometryBasePointer[2 * geometryBlockSize + triangleIndex] = mesh.vertices_z[triangleBaseIndex + 0];
    quicciMesh.geometryBasePointer[3 * geometryBlockSize + triangleIndex] = mesh.vertices_x[triangleBaseIndex + 1];
    quicciMesh.geometryBasePointer[4 * geometryBlockSize + triangleIndex] = mesh.vertices_y[triangleBaseIndex + 1];
    quicciMesh.geometryBasePointer[5 * geometryBlockSize + triangleIndex] = mesh.vertices_z[triangleBaseIndex + 1];
    quicciMesh.geometryBasePointer[6 * geometryBlockSize + triangleIndex] = mesh.vertices_x[triangleBaseIndex + 2];
    quicciMesh.geometryBasePointer[7 * geometryBlockSize + triangleIndex] = mesh.vertices_y[triangleBaseIndex + 2];
    quicciMesh.geometryBasePointer[8 * geometryBlockSize + triangleIndex] = mesh.vertices_z[triangleBaseIndex + 2];
}

__global__ void redistributeSpinOrigins(ShapeDescriptor::OrientedPoint* spinOrigins, size_t imageCount, QUICCIMesh quicciMesh) {
    assert(imageCount == gridDim.x);
    size_t imageIndex = blockIdx.x;

    size_t spinOriginsBlockSize = roundSizeToNearestCacheLine(imageCount);

    ShapeDescriptor::OrientedPoint spinOrigin = spinOrigins[imageIndex];

    quicciMesh.spinOriginsBasePointer[0 * spinOriginsBlockSize + imageIndex] = spinOrigin.vertex.x;
    quicciMesh.spinOriginsBasePointer[1 * spinOriginsBlockSize + imageIndex] = spinOrigin.vertex.y;
    quicciMesh.spinOriginsBasePointer[2 * spinOriginsBlockSize + imageIndex] = spinOrigin.vertex.z;

    quicciMesh.spinOriginsBasePointer[3 * spinOriginsBlockSize + imageIndex] = spinOrigin.normal.x;
    quicciMesh.spinOriginsBasePointer[4 * spinOriginsBlockSize + imageIndex] = spinOrigin.normal.y;
    quicciMesh.spinOriginsBasePointer[5 * spinOriginsBlockSize + imageIndex] = spinOrigin.normal.z;
}
#endif

template<unsigned int pixelDifferenceThreshold>
ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> generateQUICCImagesGPU(
        ShapeDescriptor::gpu::Mesh device_mesh,
        ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> device_descriptorOrigins,
        float supportRadius,
        ShapeDescriptor::QUICCIExecutionTimes* executionTimes)
{
#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
    auto totalExecutionTimeStart = std::chrono::steady_clock::now();

    size_t imageCount = device_descriptorOrigins.length;
    size_t meshVertexCount = device_mesh.vertexCount;
    size_t triangleCount = meshVertexCount / (size_t) 3;

    // -- Mesh Scaling --
    // The mesh is scaled to where 1 distance unit is one pixel.
    // This saves a lot of divisions when rendering the images.

    auto meshScaleTimeStart = std::chrono::steady_clock::now();

    float scaleFactor = float(spinImageWidthPixels)/supportRadius;

    ShapeDescriptor::gpu::Mesh device_editableMeshCopy = duplicateMesh(device_mesh);
    scaleQUICCIMesh<<<(meshVertexCount / 128) + 1, 128>>>(device_editableMeshCopy, scaleFactor);
    checkCudaErrors(cudaDeviceSynchronize());

    ShapeDescriptor::OrientedPoint* device_editableSpinOriginsCopy;
    checkCudaErrors(cudaMalloc(&device_editableSpinOriginsCopy, imageCount * sizeof(ShapeDescriptor::OrientedPoint)));
    checkCudaErrors(cudaMemcpy(device_editableSpinOriginsCopy, device_descriptorOrigins.content, imageCount * sizeof(ShapeDescriptor::OrientedPoint), cudaMemcpyDeviceToDevice));
    scaleQUICCISpinOrigins<<<(imageCount / 128) + 1, 128>>>(device_editableSpinOriginsCopy, imageCount, scaleFactor);
    checkCudaErrors(cudaDeviceSynchronize());

    std::chrono::milliseconds meshScaleDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - meshScaleTimeStart);

    // -- Array Restructuring --
    // In order to optimise memory access patterns, we rearrange the memory layout of the input data

    QUICCIMesh quicciMesh;
    quicciMesh.vertexCount = meshVertexCount;

    checkCudaErrors(cudaMalloc(&quicciMesh.spinOriginsBasePointer,
                               roundSizeToNearestCacheLine(imageCount) * 2 * sizeof(float3)));
    checkCudaErrors(cudaMalloc(&quicciMesh.geometryBasePointer,
                               roundSizeToNearestCacheLine(meshVertexCount) * 3 * sizeof(float3)));

    auto redistributeTimeStart = std::chrono::steady_clock::now();

    redistributeMesh<<<triangleCount, 1>>>(device_editableMeshCopy, quicciMesh);
    redistributeSpinOrigins<<<imageCount, 1>>>(device_editableSpinOriginsCopy, imageCount, quicciMesh);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    std::chrono::milliseconds redistributeDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - redistributeTimeStart);

    // -- Descriptor Array Allocation and Initialisation --
    size_t imageSequenceSize = imageCount * sizeof(ShapeDescriptor::QUICCIDescriptor);

    ShapeDescriptor::QUICCIDescriptor* device_descriptors;
    checkCudaErrors(cudaMalloc(&device_descriptors, imageSequenceSize));

    ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors;
    descriptors.content = device_descriptors;
    descriptors.length = imageCount;

    // -- Descriptor Generation --

    auto generationStart = std::chrono::steady_clock::now();

    // Warning: kernel assumes the grid dimensions are equivalent to imageCount.
    generateQuickIntersectionCountChangeImage<pixelDifferenceThreshold> <<<imageCount, RASTERISATION_WARP_SIZE>>> (device_descriptors, quicciMesh);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    std::chrono::milliseconds generationDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - generationStart);

    // -- Cleanup --

    ShapeDescriptor::free(device_editableMeshCopy);
    cudaFree(quicciMesh.spinOriginsBasePointer);
    cudaFree(quicciMesh.geometryBasePointer);
    cudaFree(device_editableSpinOriginsCopy);

    std::chrono::milliseconds totalExecutionDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - totalExecutionTimeStart);

    if(executionTimes != nullptr) {
        executionTimes->totalExecutionTimeSeconds = double(totalExecutionDuration.count()) / 1000.0;
        executionTimes->generationTimeSeconds = double(generationDuration.count()) / 1000.0;
        executionTimes->meshScaleTimeSeconds = double(meshScaleDuration.count()) / 1000.0;
        executionTimes->redistributionTimeSeconds = double(redistributeDuration.count()) / 1000.0;
    }

    return descriptors;
#else
    throw std::runtime_error(ShapeDescriptor::cudaMissingErrorMessage);
#endif
}

ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> ShapeDescriptor::generateQUICCImages(
        ShapeDescriptor::gpu::Mesh device_mesh,
        ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> device_descriptorOrigins,
        float supportRadius,
        ShapeDescriptor::QUICCIExecutionTimes* executionTimes) {
    return generateQUICCImagesGPU<1>(device_mesh, device_descriptorOrigins, supportRadius, executionTimes);
}

ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> ShapeDescriptor::generatePartialityResistantQUICCImages(
        ShapeDescriptor::gpu::Mesh device_mesh,
        ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> device_descriptorOrigins,
        float supportRadius,
        ShapeDescriptor::QUICCIExecutionTimes* executionTimes) {
    return generateQUICCImagesGPU<1>(device_mesh, device_descriptorOrigins, supportRadius, executionTimes);
}