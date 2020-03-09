#pragma once

#include <spinImage/cpu/index/types/Index.h>
#include <spinImage/cpu/index/types/IndexEntry.h>
#include <spinImage/cpu/types/QuiccImage.h>

std::vector<IndexEntry> queryIndex(Index &index, const QuiccImage &queryImage, unsigned int resultCount);

