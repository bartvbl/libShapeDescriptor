#pragma once

#include <spinImage/cpu/index/types/Index.h>
#include <string>
#include <spinImage/cpu/index/types/NodeBlock.h>

namespace SpinImage {
    namespace index {
        namespace io {
            Index readIndex(std::experimental::filesystem::path indexDirectory);

            void writeIndex(const Index& index, std::experimental::filesystem::path indexDirectory);

            NodeBlock* readNodeBlock(const std::string &blockID, const std::experimental::filesystem::path &indexRootDirectory);

            void writeNodeBlock(NodeBlock *block, const std::experimental::filesystem::path &indexRootDirectory);
        }
    }
}