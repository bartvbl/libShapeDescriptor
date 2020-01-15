#pragma once

#include "IntermediateNode.h"
#include "LeafNode.h"

union Node {
    IntermediateNode intermediate;
    LeafNode leaf;
};