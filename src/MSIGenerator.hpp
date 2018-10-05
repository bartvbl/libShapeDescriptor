#pragma once

#include <algorithm>

#include "shapeSearch/geom.hpp"
#include "shapeSearch/OBJLoader.h"
#include "shapeSearch/constants.h"
#include "shapeSearch/arrayTypes.hpp"


typedef struct PrecalculatedSettings {
	float alignmentProjection_n_ax;
	float alignmentProjection_n_ay;
	float alignmentProjection_n_bx;
	float alignmentProjection_n_bz;
} PrecalculatedSettings;

typedef struct RasterisationSettings {
	float3 spinImageVertex;
	float3 spinImageNormal;
	int vertexIndexIndex;
	int spinImageWidthPixels;

	Mesh mesh;
} RasterisationSettings;

void generateQSI(array<unsigned int> descriptor, RasterisationSettings settings);
void generateMSI(array<unsigned int> MSIDescriptor, array<unsigned int> QSIDescriptor, RasterisationSettings settings);

// utility functions
float3 transformCoordinate(float3 vertex, float3 spinImageVertex, float3 spinImageNormal);
Mesh scaleMesh(Mesh &model, Mesh &scaledModel, float spinImagePixelSize);


void computeMSI_risingHorizontal(array<unsigned int> MSIDescriptor, array<unsigned int> QSIDescriptor);
void computeMSI_fallingHorizontal(array<unsigned int> MSIDescriptor, array<unsigned int> QSIDescriptor);