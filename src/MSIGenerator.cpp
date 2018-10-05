#include "shapeSearch/MSIGenerator.hpp"
#include "SpinImageSizeCalculator.h"

float transformNormalX(PrecalculatedSettings pre_settings, float3 spinImageNormal)
{
	return pre_settings.alignmentProjection_n_ax * spinImageNormal.x + pre_settings.alignmentProjection_n_ay * spinImageNormal.y;
}

float2 alignWithPositiveX(float2 midLineDirection, float2 vertex)
{
	float2 transformed{};
	transformed.x = midLineDirection.x * vertex.x + midLineDirection.y * vertex.y;
	transformed.y = -midLineDirection.y * vertex.x + midLineDirection.x * vertex.y;
	return transformed;
}

PrecalculatedSettings calculateRotationSettings(float3 spinImageNormal) {
	PrecalculatedSettings pre_settings{};

	float2 sineCosineAlpha = normalize(make_float2(spinImageNormal.x, spinImageNormal.y));

	bool is_n_a_not_zero = !((abs(spinImageNormal.x) < MAX_EQUIVALENCE_ROUNDING_ERROR) && (abs(spinImageNormal.y) < MAX_EQUIVALENCE_ROUNDING_ERROR));

	if (is_n_a_not_zero)
	{
		pre_settings.alignmentProjection_n_ax = sineCosineAlpha.x;
		pre_settings.alignmentProjection_n_ay = sineCosineAlpha.y;
	}
	else
	{
		pre_settings.alignmentProjection_n_ax = 1;
		pre_settings.alignmentProjection_n_ay = 0;
	}

	float transformedNormalX = transformNormalX(pre_settings, spinImageNormal);

	float2 sineCosineBeta = normalize(make_float2(transformedNormalX, spinImageNormal.z));

	bool is_n_b_not_zero = !((abs(transformedNormalX) < MAX_EQUIVALENCE_ROUNDING_ERROR) && (abs(spinImageNormal.z) < MAX_EQUIVALENCE_ROUNDING_ERROR));

	if (is_n_b_not_zero)
	{
		pre_settings.alignmentProjection_n_bx = sineCosineBeta.x;
		pre_settings.alignmentProjection_n_bz = sineCosineBeta.y;
	}
	else
	{
		pre_settings.alignmentProjection_n_bx = 1;
		pre_settings.alignmentProjection_n_bz = 0;
	}

	return pre_settings;
}

float3 transformCoordinate(float3 vertex, float3 spinImageVertex, float3 spinImageNormal)
{
	PrecalculatedSettings spinImageSettings = calculateRotationSettings(spinImageNormal);
	float3 transformedCoordinate = vertex - spinImageVertex;

	float initialTransformedX = transformedCoordinate.x;
	transformedCoordinate.x = spinImageSettings.alignmentProjection_n_ax * transformedCoordinate.x + spinImageSettings.alignmentProjection_n_ay * transformedCoordinate.y;
	transformedCoordinate.y = -spinImageSettings.alignmentProjection_n_ay * initialTransformedX + spinImageSettings.alignmentProjection_n_ax * transformedCoordinate.y;

	initialTransformedX = transformedCoordinate.x;
	transformedCoordinate.x = spinImageSettings.alignmentProjection_n_bz * transformedCoordinate.x - spinImageSettings.alignmentProjection_n_bx * transformedCoordinate.z;
	transformedCoordinate.z = spinImageSettings.alignmentProjection_n_bx * initialTransformedX + spinImageSettings.alignmentProjection_n_bz * transformedCoordinate.z;

	return transformedCoordinate;
}

void rasteriseTriangle(array<unsigned int> descriptor, float3 vertices[3], RasterisationSettings settings)
{
	vertices[0] = transformCoordinate(vertices[0], settings.spinImageVertex, settings.spinImageNormal);
	vertices[1] = transformCoordinate(vertices[1], settings.spinImageVertex, settings.spinImageNormal);
	vertices[2] = transformCoordinate(vertices[2], settings.spinImageVertex, settings.spinImageNormal);

	float3 minVector = { 0, 0, 0 };
	float3 midVector = { 0, 0, 0 };
	float3 maxVector = { 0, 0, 0 };

	float3 deltaMinMid = { 0, 0, 0 };
	float3 deltaMidMax = { 0, 0, 0 };
	float3 deltaMinMax = { 0, 0, 0 };

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

	deltaMinMid = midVector - minVector;
	deltaMidMax = maxVector - midVector;
	deltaMinMax = maxVector - minVector;

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

	float centreLineFactor = deltaMinMid.z / deltaMinMax.z;
	float2 centreLineDelta = make_float2(deltaMinMax.x, deltaMinMax.y) * centreLineFactor;
	float2 centreLineDirection = centreLineDelta - make_float2(deltaMinMid.x, deltaMinMid.y);
	float2 centreDirection = normalize(centreLineDirection);

	minXY = alignWithPositiveX(centreDirection, make_float2(minVector.x, minVector.y));
	midXY = alignWithPositiveX(centreDirection, make_float2(midVector.x, midVector.y));
	maxXY = alignWithPositiveX(centreDirection, make_float2(maxVector.x, maxVector.y));

	deltaMinMidXY = midXY - minXY;
	deltaMidMaxXY = maxXY - midXY;
	deltaMinMaxXY = maxXY - minXY;

	minPixels = int(floor(minVector.z));
	maxPixels = int(floor(maxVector.z));

	minPixels = clamp(minPixels, (-settings.spinImageWidthPixels / 2), (settings.spinImageWidthPixels / 2) - 1);
	maxPixels = clamp(maxPixels, (-settings.spinImageWidthPixels / 2), (settings.spinImageWidthPixels / 2) - 1);

	int jobCount = maxPixels - minPixels;

	if (jobCount == 0) {
		return;
	}

	jobCount++;
	jobCount = std::min(minPixels + jobCount, int(settings.spinImageWidthPixels / 2)) - minPixels;

	for (int jobID = 0; jobID < jobCount; jobID++)
	{
		float jobMinVectorZ;
		float jobMidVectorZ;
		float jobDeltaMinMidZ;
		float jobDeltaMidMaxZ;
		float jobShortDeltaVectorZ;
		float jobShortVectorStartZ;
		float2 jobMinXY = minXY;
		float2 jobMidXY = midXY;
		float2 jobDeltaMinMidXY = deltaMinMidXY;
		float2 jobDeltaMidMaxXY = deltaMidMaxXY;
		float2 jobShortVectorStartXY;
		float2 jobShortTransformedDelta;

		int jobMinYPixels = minPixels;
		int jobPixelY = jobMinYPixels + jobID;

		jobMinVectorZ = minVector.z;
		jobMidVectorZ = midVector.z;

		jobDeltaMinMidZ = deltaMinMid.z;
		jobDeltaMidMaxZ = deltaMidMax.z;

		if (float(jobPixelY) <= jobMidVectorZ)
		{
			jobShortVectorStartXY = jobMinXY;
			jobShortVectorStartZ = jobMinVectorZ;
			jobShortDeltaVectorZ = jobDeltaMinMidZ;
			jobShortTransformedDelta = jobDeltaMinMidXY;
		}
		else
		{
			jobShortVectorStartXY = jobMidXY;
			jobShortVectorStartZ = jobMidVectorZ;
			jobShortDeltaVectorZ = jobDeltaMidMaxZ;
			jobShortTransformedDelta = jobDeltaMidMaxXY;
		}

		float jobZLevel = float(jobPixelY);
		float jobLongDistanceInTriangle = jobZLevel - jobMinVectorZ;
		float jobLongInterpolationFactor = jobLongDistanceInTriangle / deltaMinMax.z;
		float jobShortDistanceInTriangle = jobZLevel - jobShortVectorStartZ;
		float jobShortInterpolationFactor = (jobShortDeltaVectorZ == 0) ? 1.0f : jobShortDistanceInTriangle / jobShortDeltaVectorZ;

		int jobPixelYCoordinate = jobPixelY + (settings.spinImageWidthPixels / 2);

		if (jobLongDistanceInTriangle > 0 && jobShortDistanceInTriangle > 0)
		{
			float jobIntersectionY = jobMinXY.y + (jobLongInterpolationFactor * deltaMinMaxXY.y);
			float jobIntersection1X = jobShortVectorStartXY.x + (jobShortInterpolationFactor * jobShortTransformedDelta.x);
			float jobIntersection2X = jobMinXY.x + (jobLongInterpolationFactor * deltaMinMaxXY.x);

			float jobIntersection1Distance = length(make_float2(jobIntersection1X, jobIntersectionY));
			float jobIntersection2Distance = length(make_float2(jobIntersection2X, jobIntersectionY));

			bool jobHasDoubleIntersection = (jobIntersection1X * jobIntersection2X) < 0;

			float jobDoubleIntersectionDistance = abs(jobIntersectionY);

			float jobMinDistance = jobIntersection1Distance < jobIntersection2Distance ? jobIntersection1Distance : jobIntersection2Distance;
			float jobMaxDistance = jobIntersection1Distance > jobIntersection2Distance ? jobIntersection1Distance : jobIntersection2Distance;

			unsigned int jobRowStartPixels = unsigned(floor(jobMinDistance));
			unsigned int jobRowEndPixels = unsigned(floor(jobMaxDistance));

			jobRowStartPixels = std::min((unsigned int)settings.spinImageWidthPixels, std::max(unsigned(0), unsigned(jobRowStartPixels)));
			jobRowEndPixels = std::min((unsigned int)settings.spinImageWidthPixels, unsigned(jobRowEndPixels));

			size_t jobSpinImageBaseIndex = jobPixelYCoordinate * size_t(settings.spinImageWidthPixels);

			if (jobHasDoubleIntersection)
			{
				int jobDoubleIntersectionStartPixels = int(floor(jobDoubleIntersectionDistance));

				for (int jobX = jobDoubleIntersectionStartPixels; jobX < jobRowStartPixels; jobX++)
				{
					size_t jobPixelIndex = jobSpinImageBaseIndex + jobX;
					descriptor.content[jobPixelIndex] += 2;
					//std::cout << jobX << ", " << jobPixelYCoordinate << ": 2" << std::endl;
				}
			}

			for (int jobX = jobRowStartPixels; jobX < jobRowEndPixels; jobX++)
			{
				size_t jobPixelIndex = jobSpinImageBaseIndex + jobX;
				descriptor.content[jobPixelIndex] += 1;
				//std::cout << jobX << ", " << jobPixelYCoordinate << ": 1" << std::endl;
			}
		}
	}
}

Mesh scaleMesh(Mesh &model, Mesh &scaledModel, float spinImagePixelSize)
{
    scaledModel.indexCount = model.indexCount;
    scaledModel.indices = model.indices;
    scaledModel.normals = model.normals;
    scaledModel.vertexCount = model.vertexCount;

    scaledModel.vertices = new float3[model.vertexCount];

    for (int i = 0; i < model.vertexCount; i++) {
        scaledModel.vertices[i].x = model.vertices[i].x / spinImagePixelSize;
        scaledModel.vertices[i].y = model.vertices[i].y / spinImagePixelSize;
        scaledModel.vertices[i].z = model.vertices[i].z / spinImagePixelSize;
    }

    return model;
}



void generateQSI(array<unsigned int> descriptor, RasterisationSettings settings)
{
	// Reset the output descriptor
	std::fill(descriptor.content, descriptor.content + descriptor.length, 0);

	//#pragma omp parallel for num_threads(8)
	for (int triangleIndex = 0; triangleIndex < settings.mesh.indexCount / 3; triangleIndex += 1)
	{
		float3 vertices[3];

		size_t threadTriangleIndex0 = static_cast<size_t>(3 * triangleIndex);
		size_t threadTriangleIndex1 = static_cast<size_t>(3 * triangleIndex + 1);
		size_t threadTriangleIndex2 = static_cast<size_t>(3 * triangleIndex + 2);

		vertices[0] = settings.mesh.vertices[settings.mesh.indices[threadTriangleIndex0]];
		vertices[1] = settings.mesh.vertices[settings.mesh.indices[threadTriangleIndex1]];
		vertices[2] = settings.mesh.vertices[settings.mesh.indices[threadTriangleIndex2]];

		rasteriseTriangle(descriptor, vertices, settings);
	}
}

void computeMSI_fallingHorizontal(array<unsigned int> MSIDescriptor, array<unsigned int> QSIDescriptor) {
	for (int y = 0; y < spinImageWidthPixels; y++)
	{
		for (int x = 0; x < spinImageWidthPixels - 1; x++)
		{
			MSIDescriptor.content[y * spinImageWidthPixels + x] = QSIDescriptor.content[y * spinImageWidthPixels + x] > QSIDescriptor.content[y * spinImageWidthPixels + x + 1] ? 1 : 0;
		}
	}
}

void computeMSI_risingHorizontal(array<unsigned int> MSIDescriptor, array<unsigned int> QSIDescriptor) {
	for (int y = 0; y < spinImageWidthPixels; y++)
	{
		for (int x = 0; x < spinImageWidthPixels - 1; x++)
		{
			MSIDescriptor.content[y * spinImageWidthPixels + x] = QSIDescriptor.content[y * spinImageWidthPixels + x] < QSIDescriptor.content[y * spinImageWidthPixels + x + 1] ? 1 : 0;
		}
	}
}

void generateMSI(array<unsigned int> MSIDescriptor, array<unsigned int> QSIDescriptor, RasterisationSettings settings) {
	generateQSI(QSIDescriptor, settings);

	//computeMSI_fallingHorizontal(MSIDescriptor, QSIDescriptor);
}