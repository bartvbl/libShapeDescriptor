#pragma once

#include <experimental/filesystem>
#include <spinImage/cpu/index/types/Index.h>
#include "NodeBlockCache.h"

namespace SpinImage {
    namespace index {
        Index build(
                std::experimental::filesystem::path quicciImageDumpDirectory,
                std::experimental::filesystem::path indexDumpDirectory,
                std::experimental::filesystem::path statisticsFileDumpLocation = "/none/selected"
                );
    }

    namespace debug {
        struct IndexFileDiagnostics {
            //std::chrono::duration totalIOTime;
            //std::chrono::duration totalConstructionTime;
            //std::chrono::duration total
            //unsigned long
        };
        struct BuildIndexDiagnostics {
            std::vector<IndexFileDiagnostics> fileResults;
            float cacheHitRate;
            unsigned long totalIndexedImageCount;
            unsigned long totalItermediateNodeCount;
            unsigned long totalLeafNodeCount;
            std::array<unsigned long, NODES_PER_BLOCK> nodeCountPerIndexBranch;
            float averageImageCountPerBlock;
        };
    }
}

