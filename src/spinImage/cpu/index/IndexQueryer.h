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
                if(score != rhs.score) {
                    return score < rhs.score;
                }

                return entry < rhs.entry;
            }
        };

        std::vector<QueryResult> query(Index &index, const QuiccImage &queryImage, unsigned int resultCount);
    }
}



