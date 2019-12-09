#pragma once

#include <vector>
#include "IndexEntry.h"

struct BucketNode {
    const IndexNodeID id;
    std::vector<IndexEntry> images;

    std::vector<std::array<unsigned int, (spinImageWidthPixels * spinImageWidthPixels) / 32>> quicciImages;

    BucketNode(IndexNodeID id) : id(id) {
        images.reserve(1024);
    }
};