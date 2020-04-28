#pragma once

#include <spinImage/cpu/index/types/Index.h>
#include <spinImage/cpu/types/QuiccImage.h>
#include <spinImage/cpu/index/types/IndexEntry.h>
#include "IndexQueryer.h"

namespace SpinImage {
    namespace index {
        std::vector<QueryResult> sequentialQuery(std::experimental::filesystem::path dumpDirectory, const QuiccImage &queryImage, unsigned int resultCount, unsigned int fileStartIndex, unsigned int fileEndIndex, unsigned int num_threads = 0, debug::QueryRunInfo* runInfo = nullptr);
    }
}