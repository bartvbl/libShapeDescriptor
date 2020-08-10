#pragma once

#include <string>
#include <experimental/filesystem>

#include <shapeDescriptor/libraryBuildSettings.h>
#include <shapeDescriptor/cpu/types/array.h>
#include <shapeDescriptor/common/types/methods/RICIDescriptor.h>
#include <shapeDescriptor/common/types/methods/SpinImageDescriptor.h>
#include <shapeDescriptor/common/types/methods/QUICCIDescriptor.h>

namespace ShapeDescriptor {
    namespace dump {
        void descriptors(
                ShapeDescriptor::cpu::array<ShapeDescriptor::gpu::RICIDescriptor> hostDescriptors,
                std::experimental::filesystem::path imageDestinationFile,
                bool logarithmicImage,
                unsigned int imagesPerRow);

        void descriptors(
                ShapeDescriptor::cpu::array<ShapeDescriptor::gpu::SpinImageDescriptor> hostDescriptors,
                std::experimental::filesystem::path imageDestinationFile,
                bool logarithmicImage,
                unsigned int imagesPerRow);

        void descriptors(
                ShapeDescriptor::cpu::array<ShapeDescriptor::gpu::QUICCIDescriptor> hostDescriptors,
                std::experimental::filesystem::path imageDestinationFile,
                unsigned int imagesPerRow);
    }
}