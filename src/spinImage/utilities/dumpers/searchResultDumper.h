#pragma once

#include <spinImage/libraryBuildSettings.h>
#include <spinImage/common/types/array.h>
#include <spinImage/gpu/spinImageSearcher.cuh>

namespace SpinImage {
    namespace dump {
        void searchResults(array<gpu::SpinImageSearchResults> searchResults, size_t imageCount, std::string outputFilePath);
        void searchResults(array<gpu::RadialIntersectionCountImageSearchResults> searchResults, size_t imageCount, std::string outputFilePath);
    }
}
