#pragma once

#include <stdio.h>
#include <cstdlib>

#include <cuda_runtime.h>
#include "deviceMesh.h"
#include "shapeSearch/common/types/arrayTypes.hpp"
#include <string>




// Makes it easier to switch between generation styles





//typedef unsigned short newSpinImagePixelType;
//typedef unsigned int newKernelSpinImageDataType;



const int SHORT_SINGLE_BOTH_MASK = 0x00010001;
const int SHORT_SINGLE_ONE_MASK = 0x00000001;
const int SHORT_SINGLE_FIRST_MASK = 0x00010000;

const int SHORT_DOUBLE_BOTH_MASK = 0x00020002;
const int SHORT_DOUBLE_ONE_MASK = 0x00000002;
const int SHORT_DOUBLE_FIRST_MASK = 0x00020000;


// ----- DEBUG DUMPING -----

#define _DEBUG false

#if !_DEBUG
	#define DEBUG_DUMP_ENABLED false
#else
	#define DEBUG_DUMP_ENABLED false
#endif


// ----- REDEFINE SPIN IMAGE COUNT GENERATED TO CUT DOWN ON RUNTIMES? -----

#define REQUIRE_UNIQUE_VERTICES true

// ----- OTHER STUFF -----


// In the work division kernel, this defines how many "work batches" we aim to divide the model into.
// This is a lower bound, as it will calculate the number of cubes by _volume_, requiring the number
// of cubes per side of the bounding box to be rounded up, adding a number of additional work batches.
//const unsigned int targetCubeCount = 1;

// The number of spin images per row on the output image
const unsigned int imagesPerRow = 50;

const int rowsPerOutputImage = 64;




typedef struct CubePartition {
	unsigned int* startIndices;
	unsigned int* lengths;

	float* duplicated_vertices_x; // startIndices point here. Format: [partition 0 vertices] [partition 1 vertices]
	float* duplicated_vertices_y; // Format of each [partition vertices]: [vertex 0 of triangle 0] [vertex 0 of triangle 1] ... [vertex 1 of triangle 0] [vertex 1 of triangle 1]
	float* duplicated_vertices_z;
} CubePartition;



// The maximum angle between the normal of the spin image point and the normal of a given point
// which is to be added to the spin image. Avoids adding clutter into the image.
// The paper suggests using a value of 60 here, because in their testing it gave the best results.
#define spinImageSupportAngle 60

#define spinImageCubeSizeRatio 0.5f



#define parallelPlaneMaxDotError 0.0001







// additional values: 
// - 48K shared memory per block
// - No need to worry about race conditions within blocks
// - Data type of vertex indices: int (only up to 4GB models!!)

// Convenience type definition

// CUDA settings
struct CudaSettings {
    size_t threadsPerBlock;
    size_t blocksPerGrid;
};



CudaSettings calculateCUDASettings(size_t vertexCount, cudaDeviceProp device_information);

void printMemoryUsage();