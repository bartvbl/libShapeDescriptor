#pragma once

#include <shapeDescriptor/libraryBuildSettings.h>
#include <cstddef>

namespace ShapeDescriptor {
    namespace gpu {
        template <typename ScoreType>
        struct SearchResults {
            size_t indices[SEARCH_RESULT_COUNT];
            ScoreType scores[SEARCH_RESULT_COUNT];
        };
    }
}

