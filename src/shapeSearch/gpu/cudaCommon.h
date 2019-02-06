#pragma once

#include <stdio.h>
#include <cstdlib>

#include <cuda_runtime.h>
#include "deviceMesh.h"
#include "shapeSearch/common/types/arrayTypes.hpp"
#include <string>




// Makes it easier to switch between generation styles
//#define SPIN_IMAGE_MODE MODE_FLOAT32
#define SPIN_IMAGE_MODE MODE_UNSIGNED_INT

#define MODE_UNSIGNED_INT 2
#define MODE_UNSIGNED_SHORT 3
#define MODE_FLOAT32 4




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

const unsigned int DEBUG_BUFFER_ENTRY_COUNT = (0xFFFFFFFF) / sizeof(DebugValueBufferEntry);


// ----- REDEFINE SPIN IMAGE COUNT GENERATED TO CUT DOWN ON RUNTIMES? -----

#define REQUIRE_UNIQUE_VERTICES true

// ----- OTHER STUFF -----


// We're working with triangles throughout this program
#define verticesPerFace 3

// As part of the merging process, we need to compare floating point coordinates to check equivalence.
// Since floating points have 101 reasons for not being equal to one another in any situation, we need
// to use some sort of error threshold to distinguish them.
#define maxPointEquivalenceError 0.000001

// Toggles whether vertices are checked for duplicates
#define mergeVertices false

// resolution of the created spin images.
// Their physical size depends on the size of individual cubes and is thus calculated separately.
// Limitation: can not be more than 255 due to the array of bytes 
#define spinImageWidthPixels 2048

// How many subsamples to compute in each direction per pixel per spin image
#define spinImagePixelSubsampleSide 1

// The size of the number of cubes to inspect around the one containing the spin image vertex.
#define spinImageKernelSize 3

// For all kernels, this value is used to calculate how many blocks are needed to execute the kernel.
// This value represents the number of threads per block. The number of blocks is subsequently adjusted to cover all values.
const unsigned int blockSize = 32;

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
	uintArray minCubeIndices;
	uint3 cubeCounts;
	int totalCubeCount;
	float cubeSize;

	float* duplicated_vertices_x; // startIndices point here. Format: [partition 0 vertices] [partition 1 vertices]
	float* duplicated_vertices_y; // Format of each [partition vertices]: [vertex 0 of triangle 0] [vertex 0 of triangle 1] ... [vertex 1 of triangle 0] [vertex 1 of triangle 1]
	float* duplicated_vertices_z;
	unsigned int* duplicated_triangle_indices;
	unsigned int* duplicated_min_cube_indices;
} CubePartition;



// The maximum angle between the normal of the spin image point and the normal of a given point
// which is to be added to the spin image. Avoids adding clutter into the image.
// The paper suggests using a value of 60 here, because in their testing it gave the best results.
#define spinImageSupportAngle 60

#define spinImageCubeSizeRatio 0.5f



#define parallelPlaneMaxDotError 0.0001


// Classical spin image generation constants
// Number of threads per warp in classical spin image generation
const int SPIN_IMAGE_GENERATION_WARP_SIZE = 32;
const int samplesPerSpinImageSide = spinImageWidthPixels * spinImagePixelSubsampleSide;
const int halfSpinImageSizePixels = spinImageWidthPixels / 2;




// additional values: 
// - 48K shared memory per block
// - No need to worry about race conditions within blocks
// - Data type of vertex indices: int (only up to 4GB models!!)

// Convenience type definition

// CUDA settings
typedef struct CudaSettings {
    size_t threadsPerBlock;
    size_t blocksPerGrid;
} CudaSettings;

typedef struct OutputImageSettings {
	bool enableOutputImage;
	bool enableLogImage;
	std::string imageDestinationFile;
	std::string compressedDestinationFile;
} OutputImageSettings;

CudaSettings calculateCUDASettings(size_t vertexCount, cudaDeviceProp device_information);

void printMemoryUsage();