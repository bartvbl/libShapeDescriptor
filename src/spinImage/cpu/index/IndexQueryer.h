#pragma once

#include <spinImage/cpu/index/types/Index.h>
#include <spinImage/cpu/index/types/IndexEntry.h>
#include <spinImage/cpu/types/QuiccImage.h>

namespace SpinImage {
    namespace index {
        struct QueryResult {
            IndexEntry entry;
            QuiccImage image;
        };

        std::vector<QueryResult> query(Index &index, const QuiccImage &queryImage,
                unsigned int resultCountLimit, unsigned int distanceLimit = std::numeric_limits<unsigned int>::max());
    }
}



