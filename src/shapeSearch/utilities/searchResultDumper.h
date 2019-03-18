#pragma once

#include <shapeSearch/libraryBuildSettings.h>
#include "shapeSearch/common/types/array.h"
#include <shapeSearch/gpu/spinImageSearcher.cuh>
#include <shapeSearch/cpu/spinImageSearcher.h>

void dumpSearchResults(array<ImageSearchResults> searchResults, size_t imageCount, std::string outputFilePath);
void dumpSearchResults(std::vector<std::vector<DescriptorSearchResult>> searchResults, std::string outputFilePath);