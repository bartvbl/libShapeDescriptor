#include <cassert>
#include <iostream>
#include <chrono>
#include <map>
#include <cstring>
#include <shapeDescriptor/shapeDescriptor.h>

ShapeDescriptor::cpu::float2 calculateAlphaBeta(ShapeDescriptor::cpu::float3 spinVertex, ShapeDescriptor::cpu::float3 spinNormal, ShapeDescriptor::cpu::float3 point)
{
	// Using the projective properties of the dot product, an arbitrary point
	// can be projected on to the line defined by the vertex around which the spin image is generated
	// along with its surface normal.
	// The formula I used here yields a factor representing the number of times the normal vector should
	// be added to the spin vertex to get the closest point. However, since we are only interested in
	// the distance, we can operate on the distance value directly. 
	float beta = dot(point - spinVertex, spinNormal) / dot(spinNormal, spinNormal);

	ShapeDescriptor::cpu::float3 projectedPoint = spinVertex + beta * spinNormal;
	ShapeDescriptor::cpu::float3 delta = projectedPoint - point;
	float alpha = length(delta);

	ShapeDescriptor::cpu::float2 alphabeta = {alpha, beta};

	return alphabeta;
}

// Run once for every vertex index
void createDescriptors(
        ShapeDescriptor::OrientedPoint* device_spinImageOrigins,
        ShapeDescriptor::cpu::PointCloud pointCloud,
        ShapeDescriptor::cpu::array<ShapeDescriptor::SpinImageDescriptor> descriptors,
        float oneOverSpinImagePixelWidth,
        float supportAngleCosine, 
        size_t spinImageIndex)
{
	const ShapeDescriptor::OrientedPoint spinOrigin = device_spinImageOrigins[spinImageIndex];

	const ShapeDescriptor::cpu::float3 vertex = spinOrigin.vertex;
	const ShapeDescriptor::cpu::float3 normal = spinOrigin.normal;

	for (int sampleIndex = 0; sampleIndex < pointCloud.pointCount; sampleIndex++) {
        ShapeDescriptor::cpu::float3 samplePoint = pointCloud.vertices[sampleIndex];
        ShapeDescriptor::cpu::float3 sampleNormal = pointCloud.normals[sampleIndex];

        float sampleAngleCosine = dot(sampleNormal, normal);

        if(sampleAngleCosine < supportAngleCosine) {
            // Discard the sample
            continue;
        }

        ShapeDescriptor::cpu::float2 sampleAlphaBeta = calculateAlphaBeta(vertex, normal, samplePoint);

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
            descriptors[spinImageIndex].contents[valueIndex] += (interPixelX) * (interPixelY);
        }

        if (baseSpinImageCoordinateX + 1 >= 0 &&
            baseSpinImageCoordinateX + 1 < spinImageWidthPixels &&
            baseSpinImageCoordinateY + 0 >= -halfSpinImageSizePixels &&
            baseSpinImageCoordinateY + 0 < halfSpinImageSizePixels)
        {
            size_t valueIndex = size_t(
                    (baseSpinImageCoordinateY + 0 + spinImageWidthPixels / 2) * spinImageWidthPixels +
                     baseSpinImageCoordinateX + 1);
            descriptors[spinImageIndex].contents[valueIndex] += (1.0f - interPixelX) * (interPixelY);
        }

        if (baseSpinImageCoordinateX + 1 >= 0 &&
            baseSpinImageCoordinateX + 1 < spinImageWidthPixels &&
            baseSpinImageCoordinateY + 1 >= -halfSpinImageSizePixels &&
            baseSpinImageCoordinateY + 1 < halfSpinImageSizePixels)
        {
            size_t valueIndex = size_t(
                    (baseSpinImageCoordinateY + 1 + spinImageWidthPixels / 2) * spinImageWidthPixels +
                     baseSpinImageCoordinateX + 1);
            descriptors[spinImageIndex].contents[valueIndex] += (1.0f - interPixelX) * (1.0f - interPixelY);
        }

        if (baseSpinImageCoordinateX + 0 >= 0 &&
            baseSpinImageCoordinateX + 0 < spinImageWidthPixels &&
            baseSpinImageCoordinateY + 1 >= -halfSpinImageSizePixels &&
            baseSpinImageCoordinateY + 1 < halfSpinImageSizePixels)
        {
            size_t valueIndex = size_t(
                    (baseSpinImageCoordinateY + 1 + spinImageWidthPixels / 2) * spinImageWidthPixels +
                     baseSpinImageCoordinateX + 0);
            descriptors[spinImageIndex].contents[valueIndex] += (interPixelX) * (1.0f - interPixelY);
        }
	}
}

ShapeDescriptor::cpu::array<ShapeDescriptor::SpinImageDescriptor> ShapeDescriptor::generateSpinImagesMultiRadius(
        const ShapeDescriptor::cpu::PointCloud& pointCloud,
        const ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint>& descriptorOrigins,
        const std::vector<float>& supportRadii,
        float supportAngleDegrees,
        ShapeDescriptor::SIExecutionTimes* executionTimes) {
    auto totalExecutionTimeStart = std::chrono::steady_clock::now();
    assert(supportRadii.size() == descriptorOrigins.length);

    size_t imageCount = descriptorOrigins.length;

    float supportAngleCosine = float(std::cos(supportAngleDegrees * (M_PI / 180.0)));

    // -- Initialisation --
    auto initialisationStart = std::chrono::steady_clock::now();

    ShapeDescriptor::cpu::array<ShapeDescriptor::SpinImageDescriptor> descriptors(imageCount);

    size_t bufferSize = imageCount * sizeof(ShapeDescriptor::SpinImageDescriptor);
    std::memset(descriptors.content, 0, bufferSize);

    std::chrono::milliseconds initialisationDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - initialisationStart);


    // -- Spin Image Generation --
    auto generationStart = std::chrono::steady_clock::now();

#pragma omp parallel for schedule(dynamic) default(none) shared(imageCount, descriptorOrigins, pointCloud, descriptors, supportRadii, supportAngleCosine)
    for(size_t imageIndex = 0; imageIndex < imageCount; imageIndex++) {
        createDescriptors(
                descriptorOrigins.content,
                pointCloud,
                descriptors,
                float(spinImageWidthPixels)/supportRadii.at(imageIndex),
                supportAngleCosine,
                imageIndex);
    }

    std::chrono::milliseconds generationDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - generationStart);

    std::chrono::milliseconds totalExecutionDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - totalExecutionTimeStart);

    if(executionTimes != nullptr) {
        executionTimes->totalExecutionTimeSeconds = double(totalExecutionDuration.count()) / 1000.0;
        executionTimes->initialisationTimeSeconds = double(initialisationDuration.count()) / 1000.0;
        executionTimes->generationTimeSeconds = double(generationDuration.count()) / 1000.0;
    }

    return descriptors;
}

ShapeDescriptor::cpu::array<ShapeDescriptor::SpinImageDescriptor> ShapeDescriptor::generateSpinImages(
        ShapeDescriptor::cpu::PointCloud pointCloud,
        ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> descriptorOrigins,
        float supportRadius,
        float supportAngleDegrees,
        ShapeDescriptor::SIExecutionTimes* executionTimes) {
    std::vector<float> radii(descriptorOrigins.length, supportRadius);
    return generateSpinImagesMultiRadius(pointCloud, descriptorOrigins, radii, supportAngleDegrees, executionTimes);
}


