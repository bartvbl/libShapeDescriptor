#pragma once

#include <spinImage/cpu/index/types/Index.h>
#include <spinImage/cpu/index/types/IndexEntry.h>

std::vector<IndexEntry> queryIndex(Index index, unsigned int* queryImage, unsigned int resultCount);

