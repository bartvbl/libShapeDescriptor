#pragma once

#include <spinImage/cpu/index/types/Index.h>

std::vector<IndexEntry> queryIndex(Index index, unsigned int* queryImage, unsigned int resultCount);