#pragma once

#include <spinImage/cpu/index/types/Index.h>
#include <string>
#include <spinImage/cpu/index/types/NodeBlock.h>

namespace index {
    namespace io {
        Index loadIndex(std::experimental::filesystem::path rootFile);
        void writeIndex(Index index, std::experimental::filesystem::path outDirectory);

        NodeBlock* loadNodeBlock(const std::string &blockID, const std::experimental::filesystem::path &indexRootDirectory);
        void writeNodeBlock(const NodeBlock* block, const std::experimental::filesystem::path &indexRootDirectory);
    }
}

