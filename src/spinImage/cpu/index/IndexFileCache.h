#pragma once

#include <experimental/filesystem>
#include <utility>
#include <spinImage/cpu/index/types/IndexNode.h>
#include <spinImage/cpu/index/types/BucketNode.h>
#include <list>
#include <unordered_map>
#include "IndexIO.h"

// The cached nodes are stored as pointers to avoid accidental copies being created

struct CachedIndexNode {
    bool isDirty;
    IndexNode* node;
};

struct CachedBucketNode {
    bool isDirty;
    BucketNode* node;
};

class IndexFileCache {
private:
    const std::experimental::filesystem::path indexRoot;

    // Nodes are evicted on a Least Recently Used basis
    // This is most efficiently done by using a doubly linked list
    std::list<CachedIndexNode> lruIndexNodeQueue;
    std::list<CachedBucketNode> lruBucketNodeQueue;

    // These unordered maps hold on to the nodes themselves
    std::unordered_map<IndexNodeID, std::list<CachedIndexNode>::iterator> indexNodeMap;
    std::unordered_map<IndexNodeID, std::list<CachedBucketNode>::iterator> bucketNodeMap;

    const unsigned int indexNodeCapacity;
    const unsigned int bucketNodeCapacity;
    const unsigned int fileGroupSize;

    // The Index and Bucket node types share the same counter intended for generating UUID's
    // because it allows a bucket node to be converted into an index node without any ID updates
    IndexNodeID nextNodeID;

    // Get hold of an index node. May cause another node to be ejected
    IndexNode* getIndexNode(IndexNodeID indexNodeID);
    BucketNode* getBucketNode(IndexNodeID bucketNodeID);

    // Utility function for creating new nodes
    IndexNodeID createLink(IndexNodeID parent, const unsigned int* mipmapImage, unsigned int parentLevel, unsigned int LINK_TYPE);

    // Insert new index/bucket node
    void insertIndexNode(IndexNodeID indexNodeID, IndexNode* node);
    void insertBucketNode(IndexNodeID bucketNodeID, BucketNode* node);

    // Move used node to the front of the queue
    void touchIndexNode(IndexNodeID indexNodeID);
    void touchBucketNode(IndexNodeID bucketNodeID);

    // Set the dirty flag for nodes that are modified
    void markIndexNodeDirty(IndexNodeID indexNodeID);
    void markBucketNodeDirty(IndexNodeID bucketNodeID);

    // Eject least recently used element
    void ejectLeastRecentlyUsedIndexNode();
    void ejectLeastRecentlyUsedBucketNode();
public:
    explicit IndexFileCache(std::experimental::filesystem::path indexRoot,
            const unsigned int indexNodeCapacity,
            const unsigned int bucketNodeCapacity,
            const unsigned int fileGroupSize) :
            indexRoot(std::move(indexRoot)),
            indexNodeCapacity(indexNodeCapacity),
            bucketNodeCapacity(bucketNodeCapacity),
            fileGroupSize(fileGroupSize) {

        nextNodeID = 1;

        indexNodeMap.reserve(indexNodeCapacity);
        bucketNodeMap.reserve(bucketNodeCapacity);

        // For a better cache coherence and dump file layout, we reserve
        // the first 65k ID's for top level index nodes
        for(unsigned int i = 0; i < 65536; i++) {
            createIndexNode(0, &i, 0);
        }
    }

    // The index is responsible for holding the root node, because adding index/bucket nodes to the index
    // also sometimes requires modifying the root node.
    IndexRootNode rootNode;

    // The lookup functions return const pointers to ensure the only copy of these nodes exists in the cache
    // It also ensures the cache handles any necessary changes, as nodes need to be marked as dirty
    const IndexNode* fetchIndexNode(IndexNodeID indexNodeID);
    const BucketNode* fetchBucketNode(IndexNodeID bucketNodeID);

    // There's a limited number of possible modifications to nodes
    // These functions, as a side effect, also mark the internal cache representations dirty
    // Dirty nodes are written to disk when ejected from the cache.
    // Otherwise they are simply discarded.
    IndexNodeID createBucketNode(IndexNodeID parentIndexNodeID, const unsigned int* mipmapImage, unsigned int level);
    IndexNodeID createIndexNode(IndexNodeID parentIndexNodeID, const unsigned int* mipmapImage, unsigned int level);
    void insertImageIntoBucketNode(IndexNodeID bucketNodeID, IndexEntry entry);

    // Clear the cache, write all changes to disk
    void flush();
};