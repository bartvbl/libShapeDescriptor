#pragma once

#include <experimental/filesystem>

namespace SpinImage {
    namespace utilities {
        const char* readCompressedFile(const std::experimental::filesystem::path &archiveFile, size_t* fileSizeBytes, bool enableMultithreading);
    }
}
