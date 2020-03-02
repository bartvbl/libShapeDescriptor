#pragma once

#include <spinImage/cpu/index/types/Index.h>

namespace SpinImage {
    namespace index {
        Index build(std::string quicciImageDumpDirectory, std::string indexDumpDirectory);
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

