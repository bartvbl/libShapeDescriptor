#pragma once
#include <string>

#if RICI_PIXEL_DATATYPE == DATATYPE_UNSIGNED_INT
typedef uint32_t radialIntersectionCountImagePixelType;
#else
#error Unsupported datatype has been specified for the Radial Intersection Count Image
#endif

#if SI_PIXEL_DATATYPE == DATATYPE_FLOAT
typedef float spinImagePixelType;
#else
#error Unsupported datatype has been specified for the Spin Image
#endif

#if SHAPE_CONTEXT_HORIZONTAL_SLICE_COUNT > 32
#error Due to implementation limitations, only 32 horizontal slices are supported
#endif

#ifdef _WIN32
#define M_PI 3.1415926353
#endif

#define UINTS_PER_QUICCI ((spinImageWidthPixels * spinImageWidthPixels) / 32)

namespace ShapeDescriptor {
    const std::string cudaMissingErrorMessage = "CUDA support was not enabled during compilation of libShapeDescriptor. Please use a CPU based implementation or recompile the library with CUDA support enabled.";
}
