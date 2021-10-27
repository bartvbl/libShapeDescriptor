#pragma once

#include <array>
#include <string>
#include <experimental/filesystem>
#include <shapeDescriptor/common/types/methods/QUICCIDescriptor.h>
#include <shapeDescriptor/cpu/types/array.h>

namespace ShapeDescriptor {
    struct QUICCIDescriptorFileHeader {
        std::array<char, 5> fileID;
        size_t imageCount;
        unsigned int descriptorWidthPixels;
    };

    namespace read {
        ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> QUICCIDescriptors(const std::experimental::filesystem::path &dumpFileLocation, unsigned int decompressionThreadCount = 1);
        QUICCIDescriptorFileHeader QuicciDescriptorFileHeader(const std::experimental::filesystem::path &dumpFileLocation);
    }
}

