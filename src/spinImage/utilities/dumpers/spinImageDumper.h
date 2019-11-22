#pragma once

#include <spinImage/common/types/array.h>
#include <spinImage/libraryBuildSettings.h>
#include <spinImage/cpu/types/QUICCIImages.h>

#include <string>

namespace SpinImage {
    namespace dump {
        void descriptors(
                array<radialIntersectionCountImagePixelType> hostDescriptors,
                std::string imageDestinationFile,
                bool logarithmicImage,
                unsigned int imagesPerRow);

        void descriptors(
                array<spinImagePixelType> hostDescriptors,
                std::string imageDestinationFile,
                bool logarithmicImage,
                unsigned int imagesPerRow);

        void descriptors(
                cpu::QUICCIImages hostDescriptors,
                std::string imageDestimationFile,
                unsigned int imagesPerRow);
    }
}