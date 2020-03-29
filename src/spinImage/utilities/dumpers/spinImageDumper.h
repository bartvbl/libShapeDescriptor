#pragma once

#include <spinImage/common/types/array.h>
#include <spinImage/libraryBuildSettings.h>
#include <spinImage/cpu/types/QUICCIImages.h>
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

        void descriptors(
                cpu::QUICCIImages hostDescriptors,
                std::experimental::filesystem::path imageDestinationFile,
                unsigned int imagesPerRow);

        void descriptors(
                const std::vector<QuiccImage> &hostDescriptors,
                std::experimental::filesystem::path imageDestinationFile,
                unsigned int imagesPerRow);
    }
}