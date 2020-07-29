#pragma once

#include <string>
#include <experimental/filesystem>

#include <spinImage/libraryBuildSettings.h>
#include <spinImage/cpu/types/QUICCIImages.h>
#include <spinImage/cpu/types/array.h>
#include <spinImage/gpu/types/methods/RICIDescriptor.h>
#include <spinImage/gpu/types/methods/SpinImageDescriptor.h>
#include <spinImage/gpu/types/methods/QUICCIDescriptor.h>

namespace SpinImage {
    namespace dump {
        void descriptors(
                SpinImage::cpu::array<SpinImage::gpu::RICIDescriptor> hostDescriptors,
                std::experimental::filesystem::path imageDestinationFile,
                bool logarithmicImage,
                unsigned int imagesPerRow);

        void descriptors(
                SpinImage::cpu::array<SpinImage::gpu::SpinImageDescriptor> hostDescriptors,
                std::experimental::filesystem::path imageDestinationFile,
                bool logarithmicImage,
                unsigned int imagesPerRow);

        void descriptors(
                SpinImage::cpu::array<SpinImage::gpu::QUICCIDescriptor> hostDescriptors,
                std::experimental::filesystem::path imageDestinationFile,
                unsigned int imagesPerRow);
    }
}