#pragma once

#if RICI_PIXEL_DATATYPE == DATATYPE_UNSIGNED_INT
typedef unsigned int radialIntersectionCountImagePixelType;
#else
#error Unsupported datatype has been specified for the Radial Intersection Count Image
#endif

#if SI_PIXEL_DATATYPE == DATATYPE_FLOAT
typedef float spinImagePixelType;
#else
#error Unsupported datatype has been specified for the Spin Image
#endif

#if SC_PIXEL_DATATYPE == DATATYPE_FLOAT
typedef float shapeContextBinType;
#else
#error Unsupported datatype has been specified for the 3D Shape Context
#endif

#if SHAPE_CONTEXT_HORIZONTAL_SLICE_COUNT > 32
#error Due to implementation limitations, only 32 horizontal slices are supported
#endif

#ifdef _WIN32
#define M_PI 3.1415926353
#endif

#define UINTS_PER_QUICCI ((spinImageWidthPixels * spinImageWidthPixels) / 32)