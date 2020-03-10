#pragma once

#include <experimental/filesystem>
#include <spinImage/cpu/index/types/Index.h>
#include "NodeBlockCache.h"

namespace SpinImage {
    namespace index {
        Index build(
                std::experimental::filesystem::path quicciImageDumpDirectory,
                std::experimental::filesystem::path indexDumpDirectory,
                std::experimental::filesystem::path statisticsFileDumpLocation = "/none/selected");
    }
}

