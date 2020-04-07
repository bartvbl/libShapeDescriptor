#pragma once

#include <experimental/filesystem>
#include <spinImage/cpu/index/types/Index.h>

namespace SpinImage {
    namespace index {
        Index build(
                std::experimental::filesystem::path quicciImageDumpDirectory,
                std::experimental::filesystem::path indexDumpDirectory,
                size_t cacheNodeLimit,
                size_t cacheImageLimit,
                size_t fileStartIndex,
                size_t fileEndIndex,
                bool appendToExistingIndex = false,
                std::experimental::filesystem::path statisticsFileDumpLocation = "/none/selected");
    }
}

