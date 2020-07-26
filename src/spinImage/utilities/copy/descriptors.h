#pragma once

#include <spinImage/common/types/array.h>
#include <spinImage/libraryBuildSettings.h>
#include <spinImage/cpu/types/QUICCIImages.h>

namespace SpinImage {
    namespace copy {
        array<radialIntersectionCountImagePixelType>
        RICIDescriptorsToHost(array<radialIntersectionCountImagePixelType> device_descriptors, size_t imageCount);

        array<spinImagePixelType>
        spinImageDescriptorsToHost(array<spinImagePixelType> device_descriptors, size_t imageCount);

        SpinImage::cpu::QUICCIImages
        QUICCIDescriptorsToHost(SpinImage::gpu::QUICCIImages descriptors);

        array<radialIntersectionCountImagePixelType>
        hostDescriptorsToDevice(array<radialIntersectionCountImagePixelType> hostDescriptors, size_t imageCount);

        array<spinImagePixelType>
        hostDescriptorsToDevice(array<spinImagePixelType> hostDescriptors, size_t imageCount);
    }
}