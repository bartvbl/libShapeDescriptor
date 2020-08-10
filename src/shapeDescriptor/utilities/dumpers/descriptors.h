#pragma once

#include <string>
#include <experimental/filesystem>

#include <shapeDescriptor/libraryBuildSettings.h>
#include <shapeDescriptor/cpu/types/array.h>
#include <shapeDescriptor/common/types/methods/RICIDescriptor.h>
#include <shapeDescriptor/common/types/methods/SpinImageDescriptor.h>
#include <shapeDescriptor/common/types/methods/QUICCIDescriptor.h>

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