#include <cassert>
#include "IndexFileCache.h"

template<typename CachedItemType> CachedItemType *Cache<CachedItemType>::getItemByID(size_t itemID) {
    typename std::unordered_map<IndexNodeID, typename std::list<CachedItem<CachedItemType>>::iterator>::iterator
        it = randomAccessMap.find(itemID);
    CachedItem<CachedItemType>* cachedItemEntry = nullptr;

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

    return cachedItemEntry->item;
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
        lruItemQueue.erase(leastRecentlyUsedItem);
        randomAccessMap.erase(leastRecentlyUsedItem);

        eject(leastRecentlyUsedItem);
    }

    delete leastRecentlyUsedItem.item;
}







IndexNodeID IndexFileCache::createLink(const IndexNodeID parent, const unsigned int* mipmapImage, const unsigned int parentLevel, const unsigned int LINK_TYPE) {
    IndexNodeID createdNodeID = nextNodeID;
    nextNodeID++;

    if(parentLevel == 0) {
        unsigned short mipmap = (unsigned short) *mipmapImage;
        assert(this->rootNode.links[mipmap] == ROOT_NODE_LINK_DISABLED);

        this->rootNode.links[mipmap] = createdNodeID;
        this->rootNode.linkTypes.set(mipmap, LINK_TYPE);
    } else {
        IndexNode* parentNode = getIndexNode(parent);
        const unsigned int arraySizes[4] = {0, 2, 8, 32};
        unsigned int imageArrayLength = arraySizes[parentLevel];

        parentNode->images.insert(parentNode->images.end(), mipmapImage, mipmapImage + imageArrayLength);
        parentNode->linkTypes.push_back(LINK_TYPE);
        parentNode->links.emplace_back(createdNodeID);
    }

    return createdNodeID;
}

IndexNodeID IndexFileCache::createBucketNode(const IndexNodeID parent, const unsigned int* mipmapImage, const unsigned int parentLevel) {
    IndexNodeID newBucketNodeID = createLink(parent, mipmapImage, parentLevel, INDEX_LINK_BUCKET_NODE);
    BucketNode* bucketNode = new BucketNode(newBucketNodeID);
    insertBucketNode(newBucketNodeID, bucketNode);
    markBucketNodeDirty(newBucketNodeID);

    return newBucketNodeID;
}

IndexNodeID IndexFileCache::createIndexNode(const IndexNodeID parent, const unsigned int *mipmapImage, const unsigned int parentLevel) {
    IndexNodeID newIndexNodeID = createLink(parent, mipmapImage, parentLevel, INDEX_LINK_INDEX_NODE);
    IndexNode* indexNode = new IndexNode(newIndexNodeID);
    insertIndexNode(newIndexNodeID, indexNode);
    markIndexNodeDirty(newIndexNodeID);

    return newIndexNodeID;
}


void IndexFileCache::insertImageIntoBucketNode(IndexNodeID bucketNodeID, IndexEntry entry) {
    BucketNode* bucketNode = getBucketNode(bucketNodeID);
    bucketNode->images.emplace_back(entry);
    markBucketNodeDirty(bucketNodeID);
    touchBucketNode(bucketNodeID);
}

const IndexNode *IndexFileCache::fetchIndexNode(IndexNodeID indexNodeID) {
    return getIndexNode(indexNodeID);
}

const BucketNode *IndexFileCache::fetchBucketNode(IndexNodeID bucketNodeID) {
    return getBucketNode(bucketNodeID);
}