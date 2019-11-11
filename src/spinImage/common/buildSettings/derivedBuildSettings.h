#pragma once

#if RICI_PIXEL_DATATYPE == DATATYPE_UNSIGNED_INT
typedef unsigned int radialIntersectionCountImagePixelType;
#elif RICI_PIXEL_DATATYPE == DATATYPE_UNSIGNED_SHORT
typedef unsigned short radialIntersectionCountImagePixelType;
#elif RICI_PIXEL_DATATYPE == DATATYPE_FLOAT32
typedef float radialIntersectionCountImagePixelType;
#else
#error No valid datatype has been specified for the Radial Intersection Count Image
#endif

#if SI_PIXEL_DATATYPE == DATATYPE_FLOAT
typedef float spinImagePixelType;
#else
#error No valid datatype has been specified for the Spin Image
#endif

#ifdef _WIN32
#define M_PI 3.1415926353
#endif