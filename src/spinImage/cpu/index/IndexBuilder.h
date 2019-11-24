#pragma once

#include <spinImage/cpu/index/types/Index.h>

namespace SpinImage {
    namespace index {
        Index build(std::string quicciImageDumpDirectory, std::string indexDumpDirectory);
    }
}

