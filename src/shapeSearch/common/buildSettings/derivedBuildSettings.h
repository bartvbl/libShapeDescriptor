#pragma once

#if QSI_PIXEL_DATATYPE == DATATYPE_UNSIGNED_INT
typedef unsigned int newSpinImagePixelType;
#elif QSI_PIXEL_DATATYPE == DATATYPE_UNSIGNED_SHORT
typedef unsigned short newSpinImagePixelType;
#elif QSI_PIXEL_DATATYPE == DATATYPE_FLOAT32
typedef float newSpinImagePixelType;
#else
#error No valid datatype has been specified for the Quasi Spin Image
#endif

#if SI_PIXEL_DATATYPE == DATATYPE_FLOAT
typedef float classicSpinImagePixelType;
#else
#error No valid datatype has been specified for the Spin Image
#endif

#ifdef _WIN32
#define M_PI 3.1415926353
#endif