#pragma once

#include <string>
#include <vector>
#include "IndexEntry.h"
#include "MipmapStack.h"

enum NodeType {
    INTERMEDIATE_NODE,
    LEAF_NODE
};

struct IndexNode {
    NodeType nodeType;
    std::vector<IndexNode> children;

    IndexNode(NodeType type) : nodeType(type) {}
};

struct LeafNode : IndexNode {
    std::vector<IndexEntry> images;

    LeafNode() :
        IndexNode(LEAF_NODE) {}
};

struct IntermediateNode : IndexNode {
    unsigned short nodeLevel;
    unsigned int* mipmapImage;

    IntermediateNode(unsigned short level) :
        IndexNode(INTERMEDIATE_NODE),
        nodeLevel(level) {}
};