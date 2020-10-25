#pragma once

#include <string>
#include <experimental/filesystem>
#include <shapeDescriptor/common/types/methods/QUICCIDescriptor.h>
#include <shapeDescriptor/cpu/types/array.h>

namespace ShapeDescriptor {
    namespace read {
        ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> QUICCIDescriptors(const std::experimental::filesystem::path &dumpFileLocation, unsigned int decompressionThreadCount = 1);
    }
}

