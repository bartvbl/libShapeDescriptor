#pragma once

#include <spinImage/cpu/types/BoolArray.h>

// The IntermediateNode serves as the "intermediate level" node used for culling away other branches of the index
// It does not contain any images, and its interpretation depends on its location/level in the index overall
// This massively reduces its memory footprint and complexity of interpretation.

struct IntermediateNode {
    BoolArray<256> nodeExists;
    BoolArray<256> linkTypes;

    IntermediateNode() : nodeExists(false), linkTypes(false) {}
};