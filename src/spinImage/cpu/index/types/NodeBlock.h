#pragma once

#include <array>
#include <spinImage/cpu/types/BoolArray.h>
#include <spinImage/cpu/index/types/IndexEntry.h>
#include <mutex>
#include "NodeBlockEntry.h"

class NodeBlock {
public:
    std::string identifier;
    BoolArray<NODES_PER_BLOCK> childNodeIsLeafNode = {true};
private:
    std::vector<unsigned long> directions;
    std::array<std::vector<NodeBlockEntry>, NODES_PER_BLOCK> leafNodeContents;
public:
    std::mutex blockLock;

    std::vector<NodeBlockEntry>* getNodeContentsByIndex(unsigned long direction) {
        return &leafNodeContents.at(direction);
    }

    void insert(unsigned long direction, NodeBlockEntry entry) {
        leafNodeContents.at(direction).push_back(entry);
    }

    NodeBlock() {}

    const std::vector<unsigned long>* getOutgoingEdgeIndices() const {
        return &directions;
    }
};