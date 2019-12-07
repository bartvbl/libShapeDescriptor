#pragma once

#include <experimental/filesystem>
#include <utility>
#include <spinImage/cpu/index/types/IndexNode.h>
#include <spinImage/cpu/index/types/BucketNode.h>
#include <list>
#include "IndexIO.h"

// The cached nodes are stored as pointers to avoid accidental copies being created


struct CachedIndexNode {
    unsigned long lastModificationIndex;
    bool isDirty;
    IndexNode* node;

    bool operator()(const CachedIndexNode& lhs, const CachedIndexNode& rhs) const
    {
        return lhs.lastModificationIndex < rhs.lastModificationIndex;
    }
};

struct CachedBucketNode {
    unsigned long lastModificationIndex;
    bool isDirty;
    BucketNode* node;

    bool operator()(const CachedBucketNode& lhs, const CachedBucketNode& rhs) const
    {
        return lhs.lastModificationIndex < rhs.lastModificationIndex;
    }
};

class IndexFileCache {
private:
    const std::experimental::filesystem::path indexRoot;

    // Used to determine which entries were modified last
    // Since the modification count can potentially exceed the 32-bit limit,
    // we use a 64-bit value instead.
    unsigned long modificationIndex = 0;

    // All loaded node files are stored on a Least Recently Used basis.
    // The most efficient means to keep track of the order in which nodes were modified
    // is a heap, which is a priority queue in STL.
    std::list<CachedIndexNode> indexNodeCache;
    std::list<CachedBucketNode> bucketNodeCache;

    const unsigned int indexNodeCapacity;
    const unsigned int bucketNodeCapacity;

    IndexNodeID nextIndexNodeID;
    IndexNodeID nextBucketNodeID;
public:
    explicit IndexFileCache(std::experimental::filesystem::path indexRoot,
            const unsigned int indexNodeCapacity,
            const unsigned int bucketNodeCapacity) :
            indexRoot(std::move(indexRoot)),
            indexNodeCapacity(indexNodeCapacity),
            bucketNodeCapacity(bucketNodeCapacity) {

        nextIndexNodeID = index::io::getIndexNodeCount(indexRoot) + 1;
        nextBucketNodeID = index::io::getBucketNodeCount(indexRoot) + 1;
    }

    // The index is responsible for holding the root node, because adding index/bucket nodes to the index
    // also sometimes requires modifying the root node.
    const IndexRootNode rootNode;

    // The lookup functions return const pointers to ensure the only copy of these nodes exists in the cache
    // It also ensures the cache handles any necessary changes, as nodes need to be marked as dirty
    const IndexNode* fetchIndexNode(IndexNodeID indexNodeID);
    const BucketNode* fetchBucketNode(IndexNodeID bucketNodeID);

    // There's a limited number of possible modifications to nodes
    // These functions, as a side effect, also mark the internal cache representations dirty
    // Dirty nodes are written to disk when ejected from the cache.
    // Otherwise they are simply discarded.
    IndexNodeID createBucketNode(IndexNodeID parentIndexNodeID, unsigned int level);
    IndexNodeID createIndexNode(IndexNodeID parentIndexNodeID, unsigned int level);
    IndexNodeID promoteBucketNodeToIndexNode(IndexNodeID node);
    void insertImageIntoBucketNode(IndexNodeID bucketNodeID, IndexEntry entry, unsigned int level);

    // Clear the cache, write all changes to disk
    void flush();
};