#pragma once

#include <experimental/filesystem>

namespace SpinImage {
    namespace utilities {
        std::vector<std::experimental::filesystem::path> listDirectory(const std::string& directory);
    }
}



