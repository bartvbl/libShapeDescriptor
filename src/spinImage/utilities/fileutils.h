#pragma once

#include <experimental/filesystem>

namespace SpinImage {
    namespace utilities {
        std::vector<std::experimental::filesystem::path> listDirectory(const std::string& directory);
        void writeCompressedFile(const char* buffer, size_t bufferSize, const std::experimental::filesystem::path &archiveFile);
        const char* readCompressedFile(const std::experimental::filesystem::path &archiveFile, size_t* fileSizeBytes);
    }
}



