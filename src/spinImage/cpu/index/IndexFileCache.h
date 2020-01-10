#pragma once

#include <experimental/filesystem>
#include <utility>
#include <spinImage/cpu/index/types/IndexNode.h>
#include <spinImage/cpu/index/types/BucketNode.h>
#include <list>
#include <unordered_map>
#include <spinImage/cpu/index/types/LeafNode.h>
#include "IndexIO.h"

// The Index and Bucket node types share the same counter intended for generating UUID's
// because it allows a bucket node to be converted into an index node without any ID updates
static IndexNodeID nextNodeID = 1;

// The cached nodes are stored as pointers to avoid accidental copies being created
template<typename type> struct CachedItem {
    bool isDirty;
    type* item;
};

template<typename CachedItemType> class Cache {
private:
    // Nodes are evicted on a Least Recently Used basis
    // This is most efficiently done by using a doubly linked list
    std::list<CachedItem<CachedItemType>> lruItemQueue;

    // These hash tables allow efficient fetching of nodes from the cache
    std::unordered_map<size_t, typename std::list<CachedItem<CachedItemType>>::iterator> randomAccessMap;

    const size_t itemCapacity;
protected:
    // Get hold of an item. May cause another item to be ejected
    CachedItemType* getItemByID(size_t itemID);

    // Insert an item into the cache. May cause another item to be ejected
    void insertItem(size_t itemID, CachedItemType* item);

    // Mark an item present in the cache as most recently used
    void touchItem(size_t itemID);

    // Set the dirty flag of a given item
    void markItemDirty(size_t itemID);

    void ejectLeastRecentlyUsedItem();

    virtual void eject(CachedItemType* item) = 0;
    virtual CachedItemType* load(size_t itemID) = 0;

    explicit Cache(const size_t capacity) : itemCapacity(capacity) {
        lruItemQueue.resize(capacity);
        randomAccessMap.reserve(capacity);
    }
public:
    // The lookup functions returns const pointers to ensure the only copy of these item exist in the cache
    // It also ensures the cache handles any necessary changes, as nodes need to be marked as dirty
    const CachedItemType* fetch(size_t itemID);

    // Eject all items from the cache, leave it empty
    void flush();
};

class IndexNodeCache : Cache<IndexNode> {
private:
    const std::experimental::filesystem::path indexRoot;

    // Utility function for creating new nodes
    IndexNodeID createLink(IndexNodeID parent, const unsigned int* mipmapImage, unsigned int parentLevel, unsigned int LINK_TYPE);
public:
    IndexNodeCache(const std::experimental::filesystem::path indexRootPath, const unsigned int cacheCapacity) : {

    }

    const IndexNode* fetchIndexNode(IndexNodeID indexNodeID);
    IndexNodeID createIndexNode(IndexNodeID parentIndexNodeID, const unsigned int* mipmapImage, unsigned int level);
    void splitNode(IndexNodeID indexNodeID);
};

class LeafNodeCache : Cache<LeafNode> {
private:
    const std::experimental::filesystem::path indexRoot;
};

class BucketNodeCache : Cache<BucketNode> {
private:
    const std::experimental::filesystem::path indexRoot;
    const unsigned int fileGroupSize;
public:
    const BucketNode* fetchBucketNode(IndexNodeID bucketNodeID);
    IndexNodeID createBucketNode(IndexNodeID parentIndexNodeID, const unsigned int* mipmapImage, unsigned int level);
    void insertImageIntoBucketNode(IndexNodeID bucketNodeID, IndexEntry entry);
};