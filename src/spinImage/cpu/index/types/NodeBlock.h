#pragma once

#include <array>
#include <spinImage/cpu/types/BoolArray.h>
#include <spinImage/cpu/index/types/IndexEntry.h>
#include <mutex>
#include "NodeBlockEntry.h"

class NodeBlock {
public:
    std::string identifier;
private:
    std::vector<unsigned long> directions;
    std::vector<std::vector<NodeBlockEntry>> leafNodeContents;
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

    void markChildNodeAsLeafNode(unsigned long edge, bool isLeafNode) {

    }

    bool childNodeIsLeafNode(unsigned long edge) {
        return false;
    }
};