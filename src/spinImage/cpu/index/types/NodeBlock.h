#pragma once

#include <array>
#include "Node.h"

struct NodeBlock {
    std::string identifier;
    BoolArray<256> nodeTypes = {false};
    std::array<Node, 256> contents;
};