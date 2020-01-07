#pragma once

const bool INDEX_LINK_INDEX_NODE = false;
const bool INDEX_LINK_BUCKET_NODE = true;

#include <vector>
#include <spinImage/cpu/types/BoolVector.h>
#include "Index.h"

struct IndexNode {
    const IndexNodeID id;

    std::array<IndexNodeID, 256> links = { 0xFFFFFFFFFFFFFFFFU };

    // 1 bit per image/link. 0 = index node, 1 = bucket node
    BoolVector linkTypes;

    IndexNode(IndexNodeID id) : id(id) {}
};