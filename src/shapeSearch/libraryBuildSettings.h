#pragma once

#include <shapeSearch/common/buildSettings/buildSettingsPreamble.h>


// Specifies the data type pixels should have for the different image generators.
// Because preprocessor code changes are also applied, this is implemented as a define rather than a direct typedef.
// Valid values are DATATYPE_UNSIGNED_SHORT, DATATYPE_UNSIGNED_INT, DATATYPE_FLOAT32
#define QSI_PIXEL_DATATYPE DATATYPE_UNSIGNED_INT

// Resolution of the created spin images.
// Their physical size depends on the size of individual cubes and is thus calculated separately.
// Limitation: can not be more than 255 due to the array of bytes
#define spinImageWidthPixels 2048

// By default, the quasi spin image is generated entirely in main memory. This setting forces it to generate the image
// in shared memory instead, copying it to main memory on completion instead.
#define ENABLE_SHARED_MEMORY_IMAGE false


#include <shapeSearch/common/buildSettings/derivedBuildSettings.h>