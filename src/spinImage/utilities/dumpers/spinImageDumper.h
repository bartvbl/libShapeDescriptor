#pragma once

#include <spinImage/common/types/array.h>
#include <spinImage/libraryBuildSettings.h>

#include <string>

namespace SpinImage {
    namespace dump {
        void descriptors(
                array<quasiSpinImagePixelType> hostDescriptors,
                std::string imageDestinationFile,
                bool logarithmicImage,
                unsigned int imagesPerRow);

        void descriptors(
                array<spinImagePixelType> hostDescriptors,
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