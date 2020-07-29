#pragma once

#include <spinImage/common/types/array.h>
#include <spinImage/libraryBuildSettings.h>
#include <spinImage/cpu/types/QUICCIImages.h>
#include <experimental/filesystem>

#include <string>
#include <spinImage/cpu/types/array.h>

namespace SpinImage {
    namespace dump {
        void descriptors(
                SpinImage::cpu::array<radialIntersectionCountImagePixelType> hostDescriptors,
                std::experimental::filesystem::path imageDestinationFile,
                bool logarithmicImage,
                unsigned int imagesPerRow);

        void descriptors(
                SpinImage::cpu::array<spinImagePixelType> hostDescriptors,
                std::experimental::filesystem::path imageDestinationFile,
                bool logarithmicImage,
                unsigned int imagesPerRow);

        void descriptors(
                SpinImage::cpu::QUICCIImages hostDescriptors,
                std::experimental::filesystem::path imageDestinationFile,
                unsigned int imagesPerRow);

        void descriptors(
                const std::vector<QuiccImage> &hostDescriptors,
                std::experimental::filesystem::path imageDestinationFile,
                unsigned int imagesPerRow);
    }
}