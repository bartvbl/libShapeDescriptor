#pragma once

#include <string>
#include <experimental/filesystem>
#include <shapeDescriptor/common/types/methods/QUICCIDescriptor.h>
#include <shapeDescriptor/cpu/types/array.h>

namespace SpinImage {
    namespace dump {
        namespace raw {
            void descriptors(
                const std::experimental::filesystem::path &outputDumpFile,
                const SpinImage::cpu::array<SpinImage::gpu::QUICCIDescriptor> &images);
        }
    }

}