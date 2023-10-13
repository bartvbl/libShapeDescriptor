#include <shapeDescriptor/shapeDescriptor.h>

bool ShapeDescriptor::isCUDASupportAvailable() {
#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
    return true;
#else
    return false;
#endif
}