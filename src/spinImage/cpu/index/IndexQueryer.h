#pragma once

#include <spinImage/cpu/index/types/Index.h>
#include <spinImage/cpu/index/types/IndexEntry.h>
#include <spinImage/cpu/types/QuiccImage.h>

namespace SpinImage {
    namespace index {
        struct QueryResult {
            IndexEntry entry;
            float score = 0;
            QuiccImage image;

            bool operator<(const QueryResult &rhs) const {
                if(score != rhs.score) {
                    return score < rhs.score;
                }

                return entry < rhs.entry;
            }
        };

        namespace debug {
            struct QueryRunInfo {
                double totalQueryTime = -1;
                unsigned int threadCount = 0;
                std::array<double, spinImageWidthPixels * spinImageWidthPixels> distanceTimes;

                QueryRunInfo() {
                    std::fill(distanceTimes.begin(), distanceTimes.end(), -1);
                }
            };
        }

        std::vector<QueryResult> query(Index &index, const QuiccImage &queryImage,
                unsigned int resultCountLimit, debug::QueryRunInfo* runInfo = nullptr);
    }
}



