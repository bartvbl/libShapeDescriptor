#pragma once

#include <shapeSearch/common/types/array.h>
#include <shapeSearch/libraryBuildSettings.h>

#include <string>

namespace SpinImage {
    namespace dump {
        void descriptors(
                array<newSpinImagePixelType> hostDescriptors,
                std::string imageDestinationFile,
                bool logarithmicImage,
                unsigned int imagesPerRow);

        void descriptors(
                array<classicSpinImagePixelType> hostDescriptors,
                std::string imageDestinationFile,
                bool logarithmicImage,
                unsigned int imagesPerRow);

        void compressedImages(
                array<unsigned int> compressedDescriptors,
                std::string imageDestinationFile,
                bool logarithmicImage,
                unsigned int imagesPerRow);

        void rawCompressedImages(
                array<unsigned int> compressedDescriptors,
                std::string destination,
                unsigned int imagesPerRow);
    }
}