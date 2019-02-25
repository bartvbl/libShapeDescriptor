#include "shapeSearch/gpu/types/DeviceMesh.h"
#include "shapeSearch/common/types/array.h"

array<float> compareDescriptorsComplete(array<newSpinImagePixelType> device_descriptors,
                                        array<newSpinImagePixelType> device_otherDescriptors,
                                        size_t imageCount);
array<float> compareDescriptorsElementWise(array<newSpinImagePixelType> device_descriptors,
                                           array<newSpinImagePixelType> device_otherDescriptors,
                                           size_t imageCount);