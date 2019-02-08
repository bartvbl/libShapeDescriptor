#pragma once

#if QSI_PIXEL_DATATYPE == DATATYPE_UNSIGNED_INT
typedef unsigned int newSpinImagePixelType;
#elif QSI_PIXEL_DATATYPE == DATATYPE_UNSIGNED_SHORT
typedef unsigned short newSpinImagePixelType;
#elif QSI_PIXEL_DATATYPE == DATATYPE_FLOAT32
typedef float newSpinImagePixelType;
#endif

typedef float classicSpinImagePixelType;
