#include "radialIntersectionCountImageGenerator.h"

#include <shapeDescriptor/cpu/types/Mesh.h>
#include <shapeDescriptor/cpu/types/float2.h>
#include <shapeDescriptor/cpu/types/float3.h>
#include <shapeDescriptor/libraryBuildSettings.h>
#include <shapeDescriptor/common/types/OrientedPoint.h>

#include <cassert>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <sstream>

const int RASTERISATION_WARP_SIZE = 1024;

ShapeDescriptor::cpu::float3 transformCoordinate(const ShapeDescriptor::cpu::float3 &vertex, const ShapeDescriptor::cpu::float3 &spinImageVertex, const ShapeDescriptor::cpu::float3 &spinImageNormal) {
    const ShapeDescriptor::cpu::float2 sineCosineAlpha = normalize({spinImageNormal.x, spinImageNormal.y});

    const bool is_n_a_not_zero = !((abs(spinImageNormal.x) < MAX_EQUIVALENCE_ROUNDING_ERROR) && (abs(spinImageNormal.y) < MAX_EQUIVALENCE_ROUNDING_ERROR));

    const float alignmentProjection_n_ax = is_n_a_not_zero ? sineCosineAlpha.x : 1;
    const float alignmentProjection_n_ay = is_n_a_not_zero ? sineCosineAlpha.y : 0;

	ShapeDescriptor::cpu::float3 transformedCoordinate = vertex - spinImageVertex;

	const float initialTransformedX = transformedCoordinate.x;
	transformedCoordinate.x = alignmentProjection_n_ax * transformedCoordinate.x + alignmentProjection_n_ay * transformedCoordinate.y;
	transformedCoordinate.y = -alignmentProjection_n_ay * initialTransformedX + alignmentProjection_n_ax * transformedCoordinate.y;

    const float transformedNormalX = alignmentProjection_n_ax * spinImageNormal.x + alignmentProjection_n_ay * spinImageNormal.y;

    const ShapeDescriptor::cpu::float2 sineCosineBeta = normalize({transformedNormalX, spinImageNormal.z});

    const bool is_n_b_not_zero = !((abs(transformedNormalX) < MAX_EQUIVALENCE_ROUNDING_ERROR) && (abs(spinImageNormal.z) < MAX_EQUIVALENCE_ROUNDING_ERROR));

    const float alignmentProjection_n_bx = is_n_b_not_zero ? sineCosineBeta.x : 1;
    const float alignmentProjection_n_bz = is_n_b_not_zero ? sineCosineBeta.y : 0; // discrepancy between axis here is because we are using a 2D vector on 3D axis.

	// Order matters here
	const float initialTransformedX_2 = transformedCoordinate.x;
	transformedCoordinate.x = alignmentProjection_n_bz * transformedCoordinate.x - alignmentProjection_n_bx * transformedCoordinate.z;
	transformedCoordinate.z = alignmentProjection_n_bx * initialTransformedX_2 + alignmentProjection_n_bz * transformedCoordinate.z;

	return transformedCoordinate;
}

ShapeDescriptor::cpu::float2 alignWithPositiveX(const ShapeDescriptor::cpu::float2 &midLineDirection, const ShapeDescriptor::cpu::float2 &vertex)
{
	ShapeDescriptor::cpu::float2 transformed;
	transformed.x = midLineDirection.x * vertex.x + midLineDirection.y * vertex.y;
	transformed.y = -midLineDirection.y * vertex.x + midLineDirection.x * vertex.y;
	return transformed;
}

size_t roundSizeToNearestCacheLine(size_t sizeInBytes) {
    return (sizeInBytes + 127u) & ~((size_t) 127);
}

// Note: not thread safe due to increments by 1 and 2 not being atomic
void rasteriseTriangle(
        ShapeDescriptor::RICIDescriptor* descriptors,
        ShapeDescriptor::cpu::float3 vertices[3],
        const ShapeDescriptor::cpu::float3 &spinImageVertex,
        const ShapeDescriptor::cpu::float3 &spinImageNormal,
		size_t descriptorImageIndex) {
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

    const ShapeDescriptor::cpu::float3 minVector = vertices[minIndex];
	const ShapeDescriptor::cpu::float3 midVector = vertices[midIndex];
	const ShapeDescriptor::cpu::float3 maxVector = vertices[maxIndex];

	// Calculate deltas

    const ShapeDescriptor::cpu::float3 deltaMinMid = midVector - minVector;
    const ShapeDescriptor::cpu::float3 deltaMidMax = maxVector - midVector;
    const ShapeDescriptor::cpu::float3 deltaMinMax = maxVector - minVector;

	// Horizontal triangles are most likely not to register, and cause zero divisions, so it's easier to just get rid of them.
	if (deltaMinMax.z < MAX_EQUIVALENCE_ROUNDING_ERROR)
	{
		return;
	}

	// Step 6: Calculate centre line
	const float centreLineFactor = deltaMinMid.z / deltaMinMax.z;
    const ShapeDescriptor::cpu::float2 centreLineDelta = centreLineFactor * ShapeDescriptor::cpu::float2{deltaMinMax.x, deltaMinMax.y};
    const ShapeDescriptor::cpu::float2 centreLineDirection = centreLineDelta - ShapeDescriptor::cpu::float2{deltaMinMid.x, deltaMinMid.y};
    const ShapeDescriptor::cpu::float2 centreDirection = normalize(centreLineDirection);

	// Step 7: Rotate coordinates around origin
	// From here on out, variable names follow these conventions:
	// - X: physical relative distance to closest point on intersection line
	// - Y: Distance from origin
	const ShapeDescriptor::cpu::float2 minXY = alignWithPositiveX(centreDirection, ShapeDescriptor::cpu::float2{minVector.x, minVector.y});
    const ShapeDescriptor::cpu::float2 midXY = alignWithPositiveX(centreDirection, ShapeDescriptor::cpu::float2{midVector.x, midVector.y});
    const ShapeDescriptor::cpu::float2 maxXY = alignWithPositiveX(centreDirection, ShapeDescriptor::cpu::float2{maxVector.x, maxVector.y});

    const ShapeDescriptor::cpu::float2 deltaMinMidXY = midXY - minXY;
    const ShapeDescriptor::cpu::float2 deltaMidMaxXY = maxXY - midXY;
    const ShapeDescriptor::cpu::float2 deltaMinMaxXY = maxXY - minXY;

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

    const int startRowIndex = std::max<int>(-halfHeight, minPixels);
	const int endRowIndex = std::min<int>(halfHeight - 1, maxPixels);

	for(int pixelY = startRowIndex; pixelY <= endRowIndex; pixelY++)
	{
	    // Verified: this should be <=, because it fails for the cube tests case
		const bool isBottomSection = float(pixelY) <= midVector.z;

		// Technically I can rewrite this into two separate loops
		// However, that would increase the thread divergence
		// I believe this is the best option
		const float shortDeltaVectorZ = isBottomSection ? deltaMinMid.z : deltaMidMax.z;
		const float shortVectorStartZ = isBottomSection ? minVector.z : midVector.z;
		const ShapeDescriptor::cpu::float2 shortVectorStartXY = isBottomSection ? minXY : midXY;
		const ShapeDescriptor::cpu::float2 shortTransformedDelta = isBottomSection ? deltaMinMidXY : deltaMidMaxXY;

		const float zLevel = float(pixelY);
		const float longDistanceInTriangle = zLevel - minVector.z;
        const float longInterpolationFactor = longDistanceInTriangle / deltaMinMax.z;
        const float shortDistanceInTriangle = zLevel - shortVectorStartZ;
        const float shortInterpolationFactor = (shortDeltaVectorZ == 0) ? 1.0f : shortDistanceInTriangle / shortDeltaVectorZ;
		// Set value to 1 because we want to avoid a zero division, and we define the job Z level to be at its maximum height

        const unsigned short pixelYCoordinate = (unsigned short)(pixelY + halfHeight);
		// Avoid overlap situations, only rasterise is the interpolation factors are valid
		if (longDistanceInTriangle > 0 && shortDistanceInTriangle > 0)
		{
			// y-coordinates of both interpolated values are always equal. As such we only need to interpolate that direction once.
			// They must be equal because we have aligned the direction of the horizontal-triangle plane with the x-axis.
			const float intersectionY = minXY.y + (longInterpolationFactor * deltaMinMaxXY.y);
			// The other two x-coordinates are interpolated separately.
            const float intersection1X = shortVectorStartXY.x + (shortInterpolationFactor * shortTransformedDelta.x);
            const float intersection2X = minXY.x + (longInterpolationFactor * deltaMinMaxXY.x);

            const float intersection1Distance = length(ShapeDescriptor::cpu::float2{intersection1X, intersectionY});
            const float intersection2Distance = length(ShapeDescriptor::cpu::float2{intersection2X, intersectionY});

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
			rowStartPixels = std::min<unsigned int>((unsigned int)spinImageWidthPixels, std::max<unsigned int>(0, rowStartPixels));
			rowEndPixels = std::min<unsigned int>((unsigned int)spinImageWidthPixels, rowEndPixels);

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
					descriptors[descriptorImageIndex].contents[pixelYCoordinate * spinImageWidthPixels + jobX] += 2;
				}
			}

			// It's imperative the condition of this loop is a < comparison
			for (int jobX = rowStartPixels; jobX < rowEndPixels; jobX++)
			{
				descriptors[descriptorImageIndex].contents[pixelYCoordinate * spinImageWidthPixels + jobX] += 1;
			}
		}
	}
}

void generateRadialIntersectionCountImage(
    ShapeDescriptor::RICIDescriptor* descriptors,
    ShapeDescriptor::cpu::Mesh mesh,
	size_t imageIndex,
	float scaleFactor)
{
	ShapeDescriptor::cpu::float3 spinImageVertex = mesh.vertices[imageIndex] * scaleFactor;
	ShapeDescriptor::cpu::float3 spinImageNormal = mesh.normals[imageIndex];

	const size_t triangleCount = mesh.vertexCount / 3;
	for (int triangleIndex = 0; triangleIndex < triangleCount; triangleIndex++)
	{
		ShapeDescriptor::cpu::float3 vertices[3];

        const size_t vertexComponentBlockSize = roundSizeToNearestCacheLine(triangleCount);

		vertices[0] = mesh.vertices[3 * triangleIndex + 0] * scaleFactor;
		vertices[1] = mesh.vertices[3 * triangleIndex + 1] * scaleFactor;
		vertices[2] = mesh.vertices[3 * triangleIndex + 2] * scaleFactor;

		rasteriseTriangle(descriptors, vertices, spinImageVertex, spinImageNormal, imageIndex, );
	}
}

void scaleMesh(ShapeDescriptor::cpu::Mesh &mesh, float scaleFactor) {
	for(size_t i = 0; i < mesh.vertexCount; i++) {
		mesh.vertices[i] = mesh.vertices[i] * scaleFactor;
	}    
}


ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor> ShapeDescriptor::cpu::generateRadialIntersectionCountImages(
        ShapeDescriptor::cpu::Mesh mesh,
        ShapeDescriptor::cpu::array<OrientedPoint> descriptorOrigins,
        float spinImageWidth,
        ShapeDescriptor::cpu::RICIExecutionTimes* executionTimes) {
    auto totalExecutionTimeStart = std::chrono::steady_clock::now();

    size_t imageCount = descriptorOrigins.length;
    size_t meshVertexCount = mesh.vertexCount;
    size_t triangleCount = meshVertexCount / (size_t) 3;
        
    // -- Descriptor Array Allocation and Initialisation --

    size_t descriptorBufferSize = imageCount * sizeof(ShapeDescriptor::RICIDescriptor);
    ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor> descriptors(imageCount);
	std::memset(descriptors.content, 0, descriptorBufferSize);

    // -- Descriptor Generation --

	auto generationStart = std::chrono::steady_clock::now();

	float scaleFactor = float(spinImageWidthPixels)/spinImageWidth;

	// Warning: kernel assumes the grid dimensions are equivalent to imageCount.
    #pragma omp parallel for
	for(size_t imageIndex = 0; imageIndex < descriptors.length; imageIndex++) {
		generateRadialIntersectionCountImage(descriptors.content, mesh, imageIndex, scaleFactor);
	}
	
	std::chrono::milliseconds generationDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - generationStart);


    std::chrono::milliseconds totalExecutionDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - totalExecutionTimeStart);

    if(executionTimes != nullptr) {
        executionTimes->totalExecutionTimeSeconds = double(totalExecutionDuration.count()) / 1000.0;
        executionTimes->generationTimeSeconds = double(generationDuration.count()) / 1000.0;
	}

    return descriptors;
}