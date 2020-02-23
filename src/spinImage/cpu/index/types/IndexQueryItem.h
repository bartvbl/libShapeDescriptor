#pragma once

#include <climits>
#include <utility>
// Needed due to C++'s terrible language design
#include <spinImage/cpu/index/types/Index.h>
#include "IndexEntry.h"

struct IndexQueryItem {
    bool isIndexEntry;
    IndexEntry asIndexEntry;
    std::string asIndexNodeID;
    unsigned int minDistanceScore;

    IndexQueryItem(IndexEntry indexEntry) :
        isIndexEntry(false),
        asIndexEntry(indexEntry),
        asIndexNodeID(std::string()),
        minDistanceScore(INT_MAX) {}
    IndexQueryItem(std::string nodeID) :
        isIndexEntry(true),
        asIndexEntry(0, 0),
        asIndexNodeID(std::move(nodeID)),
        minDistanceScore(INT_MAX) {}

    // For containers
    IndexQueryItem() :
        isIndexEntry(false),
        asIndexEntry(0, 0),
        asIndexNodeID(std::string()),
        minDistanceScore(INT_MAX) {}
};