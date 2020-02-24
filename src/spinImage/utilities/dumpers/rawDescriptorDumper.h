#pragma once

#include <string>
#include <spinImage/cpu/types/QUICCIImages.h>
#include <experimental/filesystem>

namespace SpinImage {
    namespace dump {
        namespace raw {
            void descriptors(
                const std::experimental::filesystem::path &outputDumpFile,
                const SpinImage::cpu::QUICCIImages &images);
        }
    }

}