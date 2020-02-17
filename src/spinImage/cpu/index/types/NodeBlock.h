#pragma once

#include <array>
#include <spinImage/cpu/types/BoolArray.h>
#include <spinImage/cpu/index/types/IndexEntry.h>
#include <spinImage/cpu/index/types/MipmapStack.h>

struct NodeBlock {
    std::string identifier;
    BoolArray<256> nodeTypes = {false};
    std::array<short, 256> contents = {-1};

    std::vector<IndexEntry> combinedIndexEntries;
    std::vector<MipMapLevel3> combinedMipmapImages;
};