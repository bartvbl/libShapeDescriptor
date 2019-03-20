#pragma once

#include <shapeSearch/libraryBuildSettings.h>
#include "shapeSearch/common/types/array.h"
#include <shapeSearch/gpu/spinImageSearcher.cuh>
#include <shapeSearch/cpu/spinImageSearcher.h>

namespace SpinImage::dump {
    void searchResults(array<ImageSearchResults> searchResults, size_t imageCount, std::string outputFilePath);
    void searchResults(std::vector<std::vector<DescriptorSearchResult>> searchResults, std::string outputFilePath);
}
