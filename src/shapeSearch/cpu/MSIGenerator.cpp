#include "MSIGenerator.hpp"
#include "SpinImageSizeCalculator.h"

float hostTransformNormalX(PrecalculatedSettings pre_settings, float3_cpu spinImageNormal)
{
	return pre_settings.alignmentProjection_n_ax * spinImageNormal.x + pre_settings.alignmentProjection_n_ay * spinImageNormal.y;
}

float2_cpu hostAlignWithPositiveX(float2_cpu midLineDirection, float2_cpu vertex)
{
	float2_cpu transformed{};
	transformed.x = midLineDirection.x * vertex.x + midLineDirection.y * vertex.y;
	transformed.y = -midLineDirection.y * vertex.x + midLineDirection.x * vertex.y;
	return transformed;
}

PrecalculatedSettings hostCalculateRotationSettings(float3_cpu spinImageNormal) {
	PrecalculatedSettings pre_settings{};

	float2_cpu sineCosineAlpha = normalize(make_float2_cpu(spinImageNormal.x, spinImageNormal.y));

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

	float transformedNormalX = hostTransformNormalX(pre_settings, spinImageNormal);

	float2_cpu sineCosineBeta = normalize(make_float2_cpu(transformedNormalX, spinImageNormal.z));

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

float3_cpu hostTransformCoordinate(float3_cpu vertex, float3_cpu spinImageVertex, float3_cpu spinImageNormal)
{
	PrecalculatedSettings spinImageSettings = hostCalculateRotationSettings(spinImageNormal);
	float3_cpu transformedCoordinate = vertex - spinImageVertex;

	float initialTransformedX = transformedCoordinate.x;
	transformedCoordinate.x = spinImageSettings.alignmentProjection_n_ax * transformedCoordinate.x + spinImageSettings.alignmentProjection_n_ay * transformedCoordinate.y;
	transformedCoordinate.y = -spinImageSettings.alignmentProjection_n_ay * initialTransformedX + spinImageSettings.alignmentProjection_n_ax * transformedCoordinate.y;

	initialTransformedX = transformedCoordinate.x;
	transformedCoordinate.x = spinImageSettings.alignmentProjection_n_bz * transformedCoordinate.x - spinImageSettings.alignmentProjection_n_bx * transformedCoordinate.z;
	transformedCoordinate.z = spinImageSettings.alignmentProjection_n_bx * initialTransformedX + spinImageSettings.alignmentProjection_n_bz * transformedCoordinate.z;

	return transformedCoordinate;
}

void hostRasteriseTriangle(array<unsigned int> descriptor, float3_cpu *vertices, CPURasterisationSettings settings)
{
	vertices[0] = hostTransformCoordinate(vertices[0], settings.spinImageVertex, settings.spinImageNormal);
	vertices[1] = hostTransformCoordinate(vertices[1], settings.spinImageVertex, settings.spinImageNormal);
	vertices[2] = hostTransformCoordinate(vertices[2], settings.spinImageVertex, settings.spinImageNormal);

	float3_cpu minVector = { 0, 0, 0 };
	float3_cpu midVector = { 0, 0, 0 };
	float3_cpu maxVector = { 0, 0, 0 };

	float3_cpu deltaMinMid = { 0, 0, 0 };
	float3_cpu deltaMidMax = { 0, 0, 0 };
	float3_cpu deltaMinMax = { 0, 0, 0 };

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

	float2_cpu minXY = { 0, 0 };
	float2_cpu midXY = { 0, 0 };
	float2_cpu maxXY = { 0, 0 };

	float2_cpu deltaMinMidXY = { 0, 0 };
	float2_cpu deltaMidMaxXY = { 0, 0 };
	float2_cpu deltaMinMaxXY = { 0, 0 };

	int minPixels = 0;
	int maxPixels = 0;

	float centreLineFactor = deltaMinMid.z / deltaMinMax.z;
	float2_cpu centreLineDelta = make_float2_cpu(deltaMinMax.x, deltaMinMax.y) * centreLineFactor;
	float2_cpu centreLineDirection = centreLineDelta - make_float2_cpu(deltaMinMid.x, deltaMinMid.y);
	float2_cpu centreDirection = normalize(centreLineDirection);

	minXY = hostAlignWithPositiveX(centreDirection, make_float2_cpu(minVector.x, minVector.y));
	midXY = hostAlignWithPositiveX(centreDirection, make_float2_cpu(midVector.x, midVector.y));
	maxXY = hostAlignWithPositiveX(centreDirection, make_float2_cpu(maxVector.x, maxVector.y));

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
		float2_cpu jobMinXY = minXY;
		float2_cpu jobMidXY = midXY;
		float2_cpu jobDeltaMinMidXY = deltaMinMidXY;
		float2_cpu jobDeltaMidMaxXY = deltaMidMaxXY;
		float2_cpu jobShortVectorStartXY;
		float2_cpu jobShortTransformedDelta;

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

			float jobIntersection1Distance = length(make_float2_cpu(jobIntersection1X, jobIntersectionY));
			float jobIntersection2Distance = length(make_float2_cpu(jobIntersection2X, jobIntersectionY));

			bool jobHasDoubleIntersection = (jobIntersection1X * jobIntersection2X) < 0;

			float jobDoubleIntersectionDistance = abs(jobIntersectionY);

			float jobMinDistance = jobIntersection1Distance < jobIntersection2Distance ? jobIntersection1Distance : jobIntersection2Distance;
			float jobMaxDistance = jobIntersection1Distance > jobIntersection2Distance ? jobIntersection1Distance : jobIntersection2Distance;

			unsigned int jobRowStartPixels = unsigned(floor(jobMinDistance));
			unsigned int jobRowEndPixels = unsigned(floor(jobMaxDistance));

			jobRowStartPixels = std::min((unsigned int)settings.spinImageWidthPixels, std::max(unsigned(0), unsigned(jobRowStartPixels)));
			jobRowEndPixels = std::min((unsigned int)settings.spinImageWidthPixels, unsigned(jobRowEndPixels));

			size_t jobSpinImageBaseIndex = size_t(settings.vertexIndexIndex) * spinImageWidthPixels * spinImageWidthPixels + jobPixelYCoordinate * size_t(settings.spinImageWidthPixels);

			if (jobHasDoubleIntersection)
			{
				int jobDoubleIntersectionStartPixels = int(floor(jobDoubleIntersectionDistance));

				for (int jobX = jobDoubleIntersectionStartPixels; jobX < jobRowStartPixels; jobX++)
				{
					size_t jobPixelIndex = jobSpinImageBaseIndex + jobX;
#pragma omp atomic
					descriptor.content[jobPixelIndex] += 2;
					//std::cout << jobX << ", " << jobPixelYCoordinate << ": 2" << std::endl;
				}
			}

			for (int jobX = jobRowStartPixels; jobX < jobRowEndPixels; jobX++)
			{
				size_t jobPixelIndex = jobSpinImageBaseIndex + jobX;
#pragma omp atomic
				descriptor.content[jobPixelIndex] += 1;
				//std::cout << jobX << ", " << jobPixelYCoordinate << ": 1" << std::endl;
			}
		}
	}
}

HostMesh hostScaleMesh(HostMesh &model, HostMesh &scaledModel, float spinImagePixelSize)
{
    scaledModel.indexCount = model.indexCount;
    scaledModel.indices = model.indices;
    scaledModel.normals = model.normals;
    scaledModel.vertexCount = model.vertexCount;

    scaledModel.vertices = new float3_cpu[model.vertexCount];

    for (int i = 0; i < model.vertexCount; i++) {
        scaledModel.vertices[i].x = model.vertices[i].x / spinImagePixelSize;
        scaledModel.vertices[i].y = model.vertices[i].y / spinImagePixelSize;
        scaledModel.vertices[i].z = model.vertices[i].z / spinImagePixelSize;
    }

    return model;
}



void hostGenerateQSI(array<unsigned int> descriptor, CPURasterisationSettings settings)
{
	for (int triangleIndex = 0; triangleIndex < settings.mesh.indexCount / 3; triangleIndex += 1)
	{
		float3_cpu vertices[3];

		size_t threadTriangleIndex0 = static_cast<size_t>(3 * triangleIndex);
		size_t threadTriangleIndex1 = static_cast<size_t>(3 * triangleIndex + 1);
		size_t threadTriangleIndex2 = static_cast<size_t>(3 * triangleIndex + 2);

		vertices[0] = settings.mesh.vertices[threadTriangleIndex0];
		vertices[1] = settings.mesh.vertices[threadTriangleIndex1];
		vertices[2] = settings.mesh.vertices[threadTriangleIndex2];

		hostRasteriseTriangle(descriptor, vertices, settings);
	}
}

array<unsigned int> hostGenerateQSIAllVertices(CPURasterisationSettings settings) {
	array<unsigned int> descriptors;
	size_t descriptorElementCount = settings.spinImageWidthPixels * settings.spinImageWidthPixels * settings.mesh.vertexCount;
	descriptors.content = new unsigned int[descriptorElementCount];
	descriptors.length = descriptorElementCount;

	// Reset the output descriptor
	std::fill(descriptors.content, descriptors.content + descriptors.length, 0);

	for(int vertex = 0; vertex < settings.mesh.vertexCount; vertex++) {
		settings.vertexIndexIndex = vertex;
		settings.spinImageVertex = settings.mesh.vertices[vertex];
		settings.spinImageNormal = settings.mesh.normals[vertex];
		hostGenerateQSI(descriptors, settings);
	}
	return descriptors;
}

void hostComputeMSI_fallingHorizontal(array<unsigned int> MSIDescriptor, array<unsigned int> QSIDescriptor) {
	for (int y = 0; y < spinImageWidthPixels; y++)
	{
		for (int x = 0; x < spinImageWidthPixels - 1; x++)
		{
			MSIDescriptor.content[y * spinImageWidthPixels + x] = QSIDescriptor.content[y * spinImageWidthPixels + x] > QSIDescriptor.content[y * spinImageWidthPixels + x + 1] ? 1 : 0;
		}
	}
}

void hostComputeMSI_risingHorizontal(array<unsigned int> MSIDescriptor, array<unsigned int> QSIDescriptor) {
	for (int y = 0; y < spinImageWidthPixels; y++)
	{
		for (int x = 0; x < spinImageWidthPixels - 1; x++)
		{
			MSIDescriptor.content[y * spinImageWidthPixels + x] = QSIDescriptor.content[y * spinImageWidthPixels + x] < QSIDescriptor.content[y * spinImageWidthPixels + x + 1] ? 1 : 0;
		}
	}
}

void hostGenerateMSI(array<unsigned int> MSIDescriptor, array<unsigned int> QSIDescriptor,
                     CPURasterisationSettings settings) {
	hostGenerateQSI(QSIDescriptor, settings);

	//hostComputeMSI_fallingHorizontal(MSIDescriptor, QSIDescriptor);
}