#pragma once

#include <spinImage/cpu/index/types/Index.h>
#include <string>
#include <spinImage/cpu/index/types/BucketNode.h>

namespace index {
    namespace io {
        Index loadIndex(std::experimental::filesystem::path rootFile);
        void writeIndex(Index index, std::experimental::filesystem::path outDirectory);

        void writeIndexNode(const std::experimental::filesystem::path& indexRootDirectory, IndexNode* node);
        void writeBucketNode(const std::experimental::filesystem::path& indexRootDirectory, BucketNode* node);

        IndexNode* readIndexNode(const std::experimental::filesystem::path& indexRootDirectory, IndexNodeID nodeID);
        BucketNode* readBucketNode(const std::experimental::filesystem::path& indexRootDirectory, IndexNodeID nodeID);
    }
}

