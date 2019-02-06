#pragma once

#include <shapeSearch/common/types/vertexDescriptors.h>

VertexDescriptors createDescriptorsNewstyle(DeviceMesh device_mesh, CubePartition device_cubePartition, cudaDeviceProp device_information, OutputImageSettings imageSettings);

struct PrecalculatedSettings {
	// Projection coefficients calculated on a spin image basis, which rotates the normal towards the x-axis.
	float alignmentProjection_n_ax;
	float alignmentProjection_n_ay;
	// Subsequent coefficients which align the normal with the positive z-axis
	float alignmentProjection_n_bx;
	float alignmentProjection_n_bz;
};

struct RasterisationSettings {
	// Projection coefficients calculated on a spin image basis, which rotates the normal towards the x-axis.
	// Subsequent coefficients which align the normal with the positive z-axis

	// Spin image coordinate
	float3 spinImageVertex;
	float3 spinImageNormal;
	int vertexIndexIndex;

	// Geometry input
	Mesh mesh; 
	CubePartition partition;
};

struct BlockRasterisationSettings {
	int cubeContentStartIndex;
	int cubeContentEndIndex;
	int cubeIndex;
};



