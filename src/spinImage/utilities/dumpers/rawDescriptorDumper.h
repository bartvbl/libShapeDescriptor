#pragma once

#include <string>
#include <spinImage/cpu/types/QUICCIImages.h>

namespace SpinImage {
    namespace dump {
        namespace raw {
            void descriptors(
                const std::string &outputDumpFile,
                const SpinImage::cpu::QUICCIImages &images);
        }
    }

}