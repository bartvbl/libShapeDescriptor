#pragma once

#include <spinImage/common/buildSettings/buildSettingsPreamble.h>


// Specifies the data type pixels should have for the different image generators.
// Because preprocessor code changes are also applied, this is implemented as a define rather than a direct typedef.
// Valid values are DATATYPE_UNSIGNED_SHORT, DATATYPE_UNSIGNED_INT, DATATYPE_FLOAT32
#define QSI_PIXEL_DATATYPE DATATYPE_UNSIGNED_INT

// Same thing, for the spin image generator. Only DATATYPE_FLOAT is supported at this time.
#define SI_PIXEL_DATATYPE DATATYPE_FLOAT

// Resolution of the created spin images.
// Their physical size depends on the size of individual cubes and is thus calculated separately.
// Limitation: can not be more than 255 due to the array of bytes
#define spinImageWidthPixels 64

// By default, the quasi spin image is generated entirely in main memory. This setting forces it to generate the image
// in shared memory instead, copying it to main memory on completion instead.
#define ENABLE_SHARED_MEMORY_IMAGE true

// In a few places we need to check for equivalence between floating point numbers.
// This is the maximum difference between two floating numbers such that they are considered equal.
#define MAX_EQUIVALENCE_ROUNDING_ERROR 0.0001

// How many search results the spinImageSearcher should generate. Higher numbers means more register/memory usage.
// Due to the implementation, this value MUST be a multiple of 32.
#define SEARCH_RESULT_COUNT 128

// Unsupported at the moment.
#define spinImageSupportAngle 60



#include <spinImage/common/buildSettings/derivedBuildSettings.h>