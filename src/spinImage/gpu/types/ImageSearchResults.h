#pragma once

#include <spinImage/libraryBuildSettings.h>
#include <cstddef>

struct ImageSearchResults {
    size_t resultIndices[SEARCH_RESULT_COUNT];
    float resultScores[SEARCH_RESULT_COUNT];
};