#pragma once

#include <spinImage/cpu/index/types/Index.h>
#include <spinImage/cpu/index/types/WeightedIndexEntry.h>
#include <spinImage/cpu/types/QuiccImage.h>

namespace SpinImage {
    namespace index {
        struct QueryResult {
            WeightedIndexEntry entry;
            QuiccImage image;
        };

        std::vector<QueryResult> query(Index &index, const QuiccImage &queryImage, unsigned int resultCount);
    }
}



