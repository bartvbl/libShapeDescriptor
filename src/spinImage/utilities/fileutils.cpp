#include "fileutils.h"
#include <fstream>

std::vector<std::experimental::filesystem::path> SpinImage::utilities::listDirectory(const std::string& directory) {
    std::vector<std::experimental::filesystem::path> foundFiles;

    for(const auto &path : std::experimental::filesystem::directory_iterator(directory)) {
        foundFiles.emplace_back(path);
    }

    return foundFiles;
}