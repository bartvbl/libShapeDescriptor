#pragma once

#include <string>
#include <filesystem>
#include <shapeDescriptor/common/types/methods/QUICCIDescriptor.h>
#include <shapeDescriptor/cpu/types/array.h>

namespace ShapeDescriptor {
    namespace dump {
        namespace raw {
            void QUICCIDescriptors(
                const std::filesystem::path &outputDumpFile,
                const ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> &images,
                unsigned int compressionThreadCount = 1);
        }
    }

}