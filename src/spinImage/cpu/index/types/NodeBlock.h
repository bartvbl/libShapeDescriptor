#pragma once

#include <array>
#include <mutex>
#include <spinImage/cpu/types/BoolArray.h>
#include <spinImage/cpu/index/types/IndexEntry.h>
#include <spinImage/cpu/index/types/MipmapStack.h>
#include "NodeBlockEntry.h"

struct NodeBlock {
    std::string identifier;
    BoolArray<NODES_PER_BLOCK> childNodeIsLeafNode = {true};
    std::array<int, NODES_PER_BLOCK> leafNodeContentsStartIndices;
    std::array<unsigned short, NODES_PER_BLOCK> leafNodeContentsLength;
    std::mutex lock;
    int freeListStartIndex = -1;

    std::vector<NodeBlockEntry> leafNodeContents;

    NodeBlock() {
        leafNodeContentsStartIndices.fill(-1);
        leafNodeContentsLength.fill(0);
    }
};