#pragma once

#include <shapeDescriptor/libraryBuildSettings.h>
#include <shapeDescriptor/gpu/spinImageSearcher.cuh>

namespace ShapeDescriptor {
    namespace dump {
        template<typename ScoreType>
        void searchResults(ShapeDescriptor::cpu::array<gpu::SearchResults<ScoreType>> searchResults, std::string outputFilePath);
    }
}
