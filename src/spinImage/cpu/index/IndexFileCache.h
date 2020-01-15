#pragma once

#include <experimental/filesystem>
#include <utility>
#include <spinImage/cpu/index/types/IndexNode.h>
#include <list>
#include <unordered_map>
#include <spinImage/cpu/index/types/LeafNode.h>
#include <spinImage/utilities/Cache.h>
#include "IndexIO.h"

// The Index and Bucket node types share the same counter intended for generating UUID's
// because it allows a bucket node to be converted into an index node without any ID updates
static IndexNodeID nextNodeID = 1;


class IndexNodeCache : Cache<IndexNode> {
private:
    const std::experimental::filesystem::path indexRoot;
public:
    IndexNodeCache(size_t capacity, const std::experimental::filesystem::path indexRootPath,
                   const unsigned int cacheCapacity) : Cache(capacity) {

    }
    IndexNodeID createIndexNode(IndexNodeID parentIndexNodeID, const unsigned int* mipmapImage, unsigned int level);
    void splitNode(IndexNodeID indexNodeID);
};

class LeafNodeCache : Cache<LeafNode> {
private:
    const std::experimental::filesystem::path indexRoot;
};