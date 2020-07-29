#pragma once

#include <spinImage/libraryBuildSettings.h>
#include <spinImage/gpu/spinImageSearcher.cuh>

namespace SpinImage {
    namespace dump {
        void searchResults(SpinImage::cpu::array<gpu::SpinImageSearchResults> searchResults, size_t imageCount, std::string outputFilePath);
        void searchResults(SpinImage::cpu::array<gpu::RadialIntersectionCountImageSearchResults> searchResults, size_t imageCount, std::string outputFilePath);
    }
}
