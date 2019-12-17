#pragma once

#include <experimental/filesystem>

namespace SpinImage {
    namespace utilities {
        std::vector<std::experimental::filesystem::path> listDirectory(std::string directory);
        void writeCompressedFile(const char* buffer, size_t bufferSize, const std::string &archiveFile, size_t fileID);
        void readCompressedFile(const char* buffer, size_t maxSize, const std::string &archiveFile, size_t fileID);
    }
}



