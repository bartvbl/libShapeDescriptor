#pragma once

#include <spinImage/libraryBuildSettings.h>
#include <cstddef>

struct SpinImageSearchResults {
    size_t resultIndices[SEARCH_RESULT_COUNT];
    float resultScores[SEARCH_RESULT_COUNT];
};

struct QuasiSpinImageSearchResults {
    size_t resultIndices[SEARCH_RESULT_COUNT];
    int resultScores[SEARCH_RESULT_COUNT];
};