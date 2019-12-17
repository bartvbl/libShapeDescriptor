#include "fileutils.h"

std::vector<std::experimental::filesystem::path> SpinImage::utilities::listDirectory(std::string directory) {
    std::vector<std::experimental::filesystem::path> foundFiles;

    for(auto &path : std::experimental::filesystem::directory_iterator(directory)) {
        foundFiles.emplace_back(path);
    }

    return foundFiles;
}

void SpinImage::utilities::writeCompressedFile(const char *buffer, size_t bufferSize, const std::string &outputArchive, size_t fileID) {

}

void SpinImage::utilities::readCompressedFile(const char *buffer, size_t maxSize, const std::string &archiveFile, size_t fileID) {

}
