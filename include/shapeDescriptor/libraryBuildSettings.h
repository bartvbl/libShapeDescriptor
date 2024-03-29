#pragma once

#include <shapeDescriptor/buildSettings/buildSettingsPreamble.h>

#define RICI_PIXEL_DATATYPE DATATYPE_UNSIGNED_INT

// Same thing, for the spin image generator. Only DATATYPE_FLOAT is supported at this time.
#define SI_PIXEL_DATATYPE DATATYPE_FLOAT

// The datatype to use for bins of the 3D Shape Context descriptor. Only DATATYPE_FLOAT is supported at this time.
#define SC_PIXEL_DATATYPE DATATYPE_FLOAT

// Resolution of the created spin images.
// Their physical size depends on the size of individual cubes and is thus calculated separately.
// Limitation: can not be more than 255 due to the array of bytes
#ifndef spinImageWidthPixels
#define spinImageWidthPixels 32
#endif

// In a few places we need to check for equivalence between floating point numbers.
// This is the maximum difference between two floating numbers such that they are considered equal.
#define MAX_EQUIVALENCE_ROUNDING_ERROR 0.0001

// How many search results the spinImageSearcher should generate. Higher numbers means more register/memory usage.
// Due to the implementation, this value MUST be a multiple of 32.
#define SEARCH_RESULT_COUNT 128

// In the paper, an early exit clause is used to significantly speed up the comparison rate of images.
// This feature can be turned on or off using this switch
#define ENABLE_RICI_COMPARISON_EARLY_EXIT true

// Descriptor size settings for the 3D shape context method
#define SHAPE_CONTEXT_HORIZONTAL_SLICE_COUNT 15
#define SHAPE_CONTEXT_LAYER_COUNT 12
#define SHAPE_CONTEXT_VERTICAL_SLICE_COUNT 11

// Descriptor size settings for the Universal Shape Context method
constexpr uint32_t USC_HORIZONTAL_SLICE_COUNT = 14;
constexpr uint32_t USC_VERTICAL_SLICE_COUNT = 14;
constexpr uint32_t USC_SLICE_COUNT = 10;

// Descriptor size settings for the Fast Point Feature Histogram (FPFH) method
#define FPFH_BINS_PER_FEATURE 11

// Number of rotations to use when computing RoPS descriptors
#define ROPS_NUM_ROTATIONS 3
#define ROPS_HISTOGRAM_BINS 5

// Select distance function to use for comparing QUICCI images
// (uncomment one of three)
#define QUICCI_DISTANCE_FUNCTION CLUTTER_RESISTANT_DISTANCE
//#define QUICCI_DISTANCE_FUNCTION HAMMING_DISTANCE
//#define QUICCI_DISTANCE_FUNCTION WEIGHTED_HAMMING_DISTANCE


#include <shapeDescriptor/buildSettings/derivedBuildSettings.h>
