#pragma once

#if RICI_PIXEL_DATATYPE == DATATYPE_UNSIGNED_INT
typedef unsigned int radialIntersectionCountImagePixelType;
#elif RICI_PIXEL_DATATYPE == DATATYPE_UNSIGNED_SHORT
typedef unsigned short radialIntersectionCountImagePixelType;
#elif RICI_PIXEL_DATATYPE == DATATYPE_FLOAT32
typedef float radialIntersectionCountImagePixelType;
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

#ifdef _WIN32
#define M_PI 3.1415926353
#endif