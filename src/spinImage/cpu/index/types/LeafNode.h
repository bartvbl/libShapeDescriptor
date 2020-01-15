#pragma once

#include <spinImage/libraryBuildSettings.h>
#include <array>
#include "MipmapStack.h"
#include "IndexEntry.h"
#include "Index.h"

struct LeafNode {
    const IndexNodeID id;

    // Reshuffling/splitting a leaf node requires information present in
    // the mipmaps of an input image. As such we need to keep images around.
    // For space efficiency, we only keep the highest level mipmap.
    // The others can be computed based on this one.
    std::vector<MipMapLevel3> imageArray;
    std::vector<IndexEntry> entries;

    LeafNode(IndexNodeID id) : id(id) {}
};