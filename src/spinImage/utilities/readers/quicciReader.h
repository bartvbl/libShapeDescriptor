#pragma once

#include <spinImage/cpu/types/QUICCIImages.h>
#include <string>

namespace SpinImage {
    namespace read {
        SpinImage::cpu::QUICCIImages QUICCImagesFromDumpFile(const std::string &dumpFileLocation);
    }
}

