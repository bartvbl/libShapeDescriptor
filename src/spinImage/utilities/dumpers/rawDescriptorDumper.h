#pragma once

#include <string>
#include <spinImage/cpu/types/QUICCIImages.h>
#include <experimental/filesystem>
#include <spinImage/gpu/types/methods/QUICCIDescriptor.h>
#include <spinImage/cpu/types/array.h>

namespace SpinImage {
    namespace dump {
        namespace raw {
            void descriptors(
                const std::experimental::filesystem::path &outputDumpFile,
                const SpinImage::cpu::array<SpinImage::gpu::QUICCIDescriptor> &images);
        }
    }

}