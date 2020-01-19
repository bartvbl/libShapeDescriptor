#pragma once

#include <experimental/filesystem>
#include <utility>
#include <list>
#include <unordered_map>
#include <spinImage/utilities/Cache.h>
#include "IndexIO.h"

class NodeBlockCache : public Cache<std::string, NodeBlock> {
private:
    const std::experimental::filesystem::path indexRoot;
    const std::experimental::filesystem::path indexNodeDirectoryRoot;
    NodeBlock* rootNode;
protected:
    void eject(NodeBlock* item) override;
    NodeBlock* load(std::string &itemID) override;
public:
    NodeBlockCache(
            size_t capacity,
            const std::experimental::filesystem::path &indexRootPath,
            NodeBlock* root)
    :   Cache(capacity),
        indexRoot(indexRootPath),
        indexNodeDirectoryRoot(indexRootPath / "nodes"),
        rootNode(root)
        {}
    void insertImage(const MipmapStack &mipmaps, const IndexEntry reference);
};