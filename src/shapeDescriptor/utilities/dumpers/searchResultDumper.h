#pragma once

#include <shapeDescriptor/libraryBuildSettings.h>
#include <shapeDescriptor/gpu/spinImageSearcher.cuh>

namespace ShapeDescriptor {
    namespace dump {
        void searchResults(ShapeDescriptor::cpu::array<gpu::SpinImageSearchResults> searchResults, size_t imageCount, std::string outputFilePath);
        void searchResults(ShapeDescriptor::cpu::array<gpu::RadialIntersectionCountImageSearchResults> searchResults, size_t imageCount, std::string outputFilePath);
    }
}
