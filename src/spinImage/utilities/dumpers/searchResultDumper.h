#pragma once

#include <spinImage/libraryBuildSettings.h>
#include <spinImage/common/types/array.h>
#include <spinImage/gpu/spinImageSearcher.cuh>
#include <spinImage/cpu/spinImageSearcher.h>

namespace SpinImage {
    namespace dump {
        void searchResults(array<ImageSearchResults> searchResults, size_t imageCount, std::string outputFilePath);
        void searchResults(std::vector<std::vector<SpinImageSearchResult>> searchResults, std::string outputFilePath);
    }
}
