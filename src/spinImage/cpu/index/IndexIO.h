#pragma once

#include <spinImage/cpu/index/types/Index.h>
#include <string>

namespace index {
    namespace io {
        Index loadIndex(std::experimental::filesystem::path rootFile);
        void writeIndex(Index index, std::experimental::filesystem::path outDirectory);

        void writeIndexNodes(const std::experimental::filesystem::path& indexRootDirectory, const std::vector<IndexNode *> &nodes, unsigned int fileGroupSize);

        IndexNode* readIndexNode(const std::experimental::filesystem::path& indexRootDirectory, IndexNodeID nodeID, unsigned int fileGroupSize);
    }
}

