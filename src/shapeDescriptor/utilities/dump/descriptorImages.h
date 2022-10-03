#pragma once

#include <string>
#include <filesystem>

#include <shapeDescriptor/libraryBuildSettings.h>
#include <shapeDescriptor/cpu/types/array.h>
#include <shapeDescriptor/common/types/methods/RICIDescriptor.h>
#include <shapeDescriptor/common/types/methods/SpinImageDescriptor.h>
#include <shapeDescriptor/common/types/methods/QUICCIDescriptor.h>

namespace ShapeDescriptor {
    namespace dump {
        void descriptors(
                ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor> hostDescriptors,
                std::filesystem::path imageDestinationFile,
                bool logarithmicImage = true,
                unsigned int imagesPerRow = 50);

        void descriptors(
                ShapeDescriptor::cpu::array<ShapeDescriptor::SpinImageDescriptor> hostDescriptors,
                std::filesystem::path imageDestinationFile,
                bool logarithmicImage = true,
                unsigned int imagesPerRow = 50);

        void descriptors(
                ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> hostDescriptors,
                std::filesystem::path imageDestinationFile,
                unsigned int imagesPerRow = 50);

        // Write an image where each channel shows a different descriptor.
        // Useful for comparing similarity of different QUICCI descriptors
        void descriptorComparisonImage(
                std::filesystem::path imageDestinationFile,
                ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> blueChannelDescriptors = {0, nullptr},
                ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> greenChannelDescriptors = {0, nullptr},
                ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> redChannelDescriptors = {0, nullptr},
                unsigned int imagesPerRow = 50);
    }
}