#pragma once

#include <array>
#include <spinImage/cpu/types/BoolArray.h>
#include <spinImage/cpu/index/types/IndexEntry.h>
#include <spinImage/cpu/index/types/MipmapStack.h>
#include "NodeBlockEntry.h"

struct NodeBlock {
    std::string identifier;
    BoolArray<256> childNodeIsLeafNode = {false};
    std::array<int, 256> leafNodeContentsStartIndices = {-1};
    std::array<unsigned short, 256> leafNodeContentsLength = {0};
    int freeListStartIndex = -1;

    std::vector<NodeBlockEntry> leafNodeContents;
};