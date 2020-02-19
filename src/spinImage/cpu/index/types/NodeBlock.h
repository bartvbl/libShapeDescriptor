#pragma once

#include <array>
#include <spinImage/cpu/types/BoolArray.h>
#include <spinImage/cpu/index/types/IndexEntry.h>
#include <spinImage/cpu/index/types/MipmapStack.h>
#include "NodeBlockEntry.h"

struct NodeBlock {
    std::string identifier;
    BoolArray<256> childNodeIsLeafNode = {false};
    std::array<int, 256> entryStartIndices = {-1};

    std::vector<NodeBlockEntry> entries;
};