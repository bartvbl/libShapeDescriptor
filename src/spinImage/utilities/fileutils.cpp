#include "fileutils.h"

std::vector<std::experimental::filesystem::path> SpinImage::utilities::listDirectory(std::string directory) {
    std::vector<std::experimental::filesystem::path> foundFiles;

    for(auto &path : std::experimental::filesystem::directory_iterator(directory)) {
        foundFiles.emplace_back(path);
    }

    return foundFiles;
}