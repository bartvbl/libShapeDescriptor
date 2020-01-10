#include <cassert>
#include "IndexFileCache.h"

template<typename CachedItemType> CachedItemType *Cache<CachedItemType>::getItemByID(size_t itemID) {
    typename std::unordered_map<IndexNodeID, typename std::list<CachedItem<CachedItemType>>::iterator>::iterator
        it = randomAccessMap.find(itemID);
    CachedItemType* cachedItemEntry = nullptr;

    if(it != randomAccessMap.end())
    {
        // Cache hit
        cachedItemEntry = it->second->item;
        touchItem(itemID);
    } else {
        // Cache miss. Load the item into the cache instead
        cachedItemEntry = load(itemID);
        insertItem(itemID, cachedItemEntry);
    }

    return cachedItemEntry;
}

template<typename CachedItemType> void Cache<CachedItemType>::flush() {
    // Perhaps not the most efficient, but this method will not be called often anyway
    while(!randomAccessMap.empty()) {
        ejectLeastRecentlyUsedItem();
    }
}

template<typename CachedItemType> void Cache<CachedItemType>::touchItem(size_t itemID) {
    // Move the desired node to the front of the LRU queue
    lruItemQueue.splice(lruItemQueue.begin(), lruItemQueue, randomAccessMap[itemID]);
}

template<typename CachedItemType> void Cache<CachedItemType>::insertItem(size_t indexNodeID, CachedItemType* item) {
    CachedItem<CachedItemType> cachedItem = {false, nullptr};
    cachedItem.item = item;

    // If cache is full, eject a node before we insert the new one
    if(lruItemQueue.size() == itemCapacity) {
        ejectLeastRecentlyUsedItem();
    }

    // When the node is inserted, it is by definition the most recently used one
    // We therefore put it in the front of the queue right away
    lruItemQueue.emplace_front(cachedItem);
    randomAccessMap[indexNodeID] = lruItemQueue.begin();
}

template<typename CachedItemType> void Cache<CachedItemType>::markItemDirty(size_t itemID) {
    typename std::unordered_map<IndexNodeID, typename std::list<CachedItem<CachedItemType>>::iterator>::iterator it = randomAccessMap.find(itemID);
    it->second->isDirty = true;
}

// The cache ends up mostly ejecting nodes. Reuse is not all that common.
// We therefore select the node that has been the least recently used, eject it,
// along with all other nodes that will end up in the same dump file.
// This will cause some often used nodes to be ejected, but should mostly reduce the
// number of times a single file is rewritten, thereby improving outflow speed of the cache.
template<typename CachedItemType> void Cache<CachedItemType>::ejectLeastRecentlyUsedItem() {
    CachedItem<CachedItemType> leastRecentlyUsedItem = lruItemQueue.back();

    if(leastRecentlyUsedItem.isDirty) {
        eject(leastRecentlyUsedItem.item);

        lruItemQueue.erase(leastRecentlyUsedItem);
        randomAccessMap.erase(leastRecentlyUsedItem);
    }

    delete leastRecentlyUsedItem.item;
}

template<typename CachedItemType>
const CachedItemType *Cache<CachedItemType>::fetch(size_t itemID) {
    return getItemByID(itemID);
}


IndexNodeID IndexFileCache::createLink(const IndexNodeID parent, const unsigned int* mipmapImage, const unsigned int parentLevel, const unsigned int LINK_TYPE) {
    IndexNodeID createdNodeID = nextNodeID;
    nextNodeID++;

    IndexNode* parentNode = fetch(parent);
    const unsigned int arraySizes[4] = {0, 2, 8, 32};
    unsigned int imageArrayLength = arraySizes[parentLevel];

    parentNode->images.insert(parentNode->images.end(), mipmapImage, mipmapImage + imageArrayLength);
    parentNode->linkTypes.push_back(LINK_TYPE);
    parentNode->links.emplace_back(createdNodeID);


    return createdNodeID;
}

IndexNodeID BucketNodeCache::createBucketNode(const IndexNodeID parent, const unsigned int* mipmapImage, const unsigned int parentLevel) {
    IndexNodeID newBucketNodeID = createLink(parent, mipmapImage, parentLevel, 0xFFFFFFFFU);
    BucketNode* bucketNode = new BucketNode(newBucketNodeID);
    insertItem(newBucketNodeID, bucketNode);
    markItemDirty(newBucketNodeID);

    return newBucketNodeID;
}

IndexNodeID IndexNodeCache::createIndexNode(const IndexNodeID parent, const unsigned int *mipmapImage, const unsigned int parentLevel) {
    IndexNodeID newIndexNodeID = createLink(parent, mipmapImage, parentLevel, 0xFFFFFFFFU);
    IndexNode* indexNode = new IndexNode(newIndexNodeID);
    insertItem(newIndexNodeID, indexNode);
    markItemDirty(newIndexNodeID);

    return newIndexNodeID;
}


void BucketNodeCache::insertImageIntoBucketNode(IndexNodeID bucketNodeID, IndexEntry entry) {
    BucketNode* bucketNode = getItemByID(bucketNodeID);
    bucketNode->images.emplace_back(entry);
    markItemDirty(bucketNodeID);
    touchItem(bucketNodeID);
}