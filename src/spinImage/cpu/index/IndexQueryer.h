#pragma once

#include <spinImage/cpu/index/types/Index.h>
#include <spinImage/cpu/types/QuiccImage.h>
#include <spinImage/cpu/index/types/IndexEntry.h>

namespace SpinImage {
    namespace index {
        struct QueryResult {
            IndexEntry entry;
            float score;

            bool operator<(const QueryResult &rhs) const {
                return score < rhs.score;
            }
        };

        std::vector<QueryResult> query(Index &index, const QuiccImage &queryImage, unsigned int resultCount);
    }
}



