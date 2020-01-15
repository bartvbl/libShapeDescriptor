#pragma once

#include <experimental/filesystem>
#include <utility>
#include <spinImage/cpu/index/types/IntermediateNode.h>
#include <list>
#include <unordered_map>
#include <spinImage/cpu/index/types/LeafNode.h>
#include <spinImage/utilities/Cache.h>
#include <spinImage/cpu/index/types/NodeBlock.h>
#include "IndexIO.h"

// The Index and Bucket node types share the same counter intended for generating UUID's
// because it allows a bucket node to be converted into an index node without any ID updates
static IndexNodeID nextNodeID = 1;


class NodeBlockCache : Cache<std::string, NodeBlock> {
private:
    const std::experimental::filesystem::path indexRoot;
public:
    NodeBlockCache(size_t capacity, const std::experimental::filesystem::path indexRootPath,
                   const unsigned int cacheCapacity) : Cache(capacity) {

    }
    IndexNodeID createIndexNode(IndexNodeID parentIndexNodeID, const unsigned int* mipmapImage, unsigned int level);
    void splitNode(IndexNodeID indexNodeID);
};