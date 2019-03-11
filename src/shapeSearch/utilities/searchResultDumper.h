#pragma once

#include <shapeSearch/libraryBuildSettings.h>
#include "shapeSearch/common/types/array.h"
#include <shapeSearch/gpu/spinImageSearcher.cuh>

void dumpSearchResults(array<ImageSearchResults> searchResults, size_t imageCount, std::string outputFilePath);