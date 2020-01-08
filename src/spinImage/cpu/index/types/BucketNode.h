#pragma once

#include <vector>
#include "IndexEntry.h"

// Bucket nodes are a "last resort" node which can expand indefinitely.
// It's meant to catch any remaining nodes that need to be added into the index
// when all previous levels are full of nodes.

struct BucketNode {
    const IndexNodeID id;
    std::vector<IndexEntry> images;

    BucketNode(IndexNodeID id) : id(id) {
        images.reserve(1024);
    }
};