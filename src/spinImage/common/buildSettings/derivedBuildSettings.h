#pragma once

#if QSI_PIXEL_DATATYPE == DATATYPE_UNSIGNED_INT
typedef unsigned int quasiSpinImagePixelType;
#elif QSI_PIXEL_DATATYPE == DATATYPE_UNSIGNED_SHORT
typedef unsigned short quasiSpinImagePixelType;
#elif QSI_PIXEL_DATATYPE == DATATYPE_FLOAT32
typedef float quasiSpinImagePixelType;
#else
#error No valid datatype has been specified for the Quasi Spin Image
#endif

#if SI_PIXEL_DATATYPE == DATATYPE_FLOAT
typedef float spinImagePixelType;
#else
#error No valid datatype has been specified for the Spin Image
#endif

#ifdef _WIN32
#define M_PI 3.1415926353
#endif