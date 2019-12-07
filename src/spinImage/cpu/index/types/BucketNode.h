#pragma once

#include <vector>
#include "IndexEntry.h"

struct BucketNode {
    const IndexNodeID id;
    std::vector<IndexEntry> images;

    BucketNode(IndexNodeID id) : id(id) {
        images.reserve(1024);
    }
};