#pragma once

#include <spinImage/common/types/array.h>
#include <spinImage/libraryBuildSettings.h>
#include <experimental/filesystem>

#include <string>

namespace SpinImage {
    namespace dump {
        void descriptors(
                array<radialIntersectionCountImagePixelType> hostDescriptors,
                std::experimental::filesystem::path imageDestinationFile,
                bool logarithmicImage,
                unsigned int imagesPerRow);

        void descriptors(
                array<spinImagePixelType> hostDescriptors,
                std::experimental::filesystem::path imageDestinationFile,
                bool logarithmicImage,
                unsigned int imagesPerRow);
    }
}