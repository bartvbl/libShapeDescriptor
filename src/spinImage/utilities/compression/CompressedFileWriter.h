#pragma once

#include <cstddef>
#include <experimental/filesystem>

namespace SpinImage {
    namespace utilities {
        void writeCompressedFile(const char* buffer, size_t bufferSize, const std::experimental::filesystem::path &archiveFile);
    }
}