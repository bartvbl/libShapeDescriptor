#pragma once

#include <vector>
#include <spinImage/cpu/index/types/NodeLinkArray.h>
#include "Index.h"

// The IndexNode serves as the "intermediate level" node used for culling away other branches of the index
// It does not contain any images, and its interpretation depends on its location/level in the index overall
// This massively reduces its memory footprint and complexity of interpretation.

struct IndexNode {
    const IndexNodeID id;

    BoolArray<256> nodeExists;
    BoolArray<256> linkTypes;

    IndexNode(IndexNodeID id) : id(id) {}
};