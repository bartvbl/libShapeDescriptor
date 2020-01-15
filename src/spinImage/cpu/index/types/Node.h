#pragma once

#include "IntermediateNode.h"
#include "LeafNode.h"

// Struct sizes (at the time of writing):
// - IntermediateNode: 64 bytes (2 x 256 bit arrays)
// - LeafNode: 48 bytes + external storage (2 std::vectors, 24 bytes each)

union Node {
    IntermediateNode intermediate;
    LeafNode leaf;
};