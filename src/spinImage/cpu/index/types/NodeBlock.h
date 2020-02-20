#pragma once

#include <array>
#include <spinImage/cpu/types/BoolArray.h>
#include <spinImage/cpu/index/types/IndexEntry.h>
#include <spinImage/cpu/index/types/MipmapStack.h>
#include "NodeBlockEntry.h"

struct NodeBlock {
    std::string identifier;
    BoolArray<NODES_PER_BLOCK> childNodeIsLeafNode = {false};
    std::array<int, NODES_PER_BLOCK> leafNodeContentsStartIndices = {-1};
    std::array<unsigned short, NODES_PER_BLOCK> leafNodeContentsLength = {0};
    int freeListStartIndex = -1;

    std::vector<NodeBlockEntry> leafNodeContents;
};