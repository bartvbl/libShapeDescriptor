#pragma once

#include <algorithm>

#include "shapeSearch/common/geom.hpp"
#include "shapeSearch/common/OBJLoader.h"
#include "constants.h"
#include "shapeSearch/common/types/arrayTypes.hpp"


struct PrecalculatedSettings {
	float alignmentProjection_n_ax;
	float alignmentProjection_n_ay;
	float alignmentProjection_n_bx;
	float alignmentProjection_n_bz;
};

struct RasterisationSettings {
	float3_cpu spinImageVertex;
	float3_cpu spinImageNormal;
	int vertexIndexIndex;
	int spinImageWidthPixels;

	HostMesh mesh;
};

void hostGenerateQSI(array<unsigned int> descriptor, RasterisationSettings settings);
array<unsigned int> hostGenerateQSIAllVertices(RasterisationSettings settings);

void hostGenerateMSI(array<unsigned int> MSIDescriptor, array<unsigned int> QSIDescriptor,
                     RasterisationSettings settings);
// utility functions
float3_cpu hostTransformCoordinate(float3_cpu vertex, float3_cpu spinImageVertex, float3_cpu spinImageNormal);


HostMesh hostScaleMesh(HostMesh &model, HostMesh &scaledModel, float spinImagePixelSize);
void hostComputeMSI_risingHorizontal(array<unsigned int> MSIDescriptor, array<unsigned int> QSIDescriptor);
void hostComputeMSI_fallingHorizontal(array<unsigned int> MSIDescriptor, array<unsigned int> QSIDescriptor);
