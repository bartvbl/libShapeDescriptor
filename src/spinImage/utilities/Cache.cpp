#include "Cache.h"

template<typename IDType, typename CachedItemType>
CachedItemType *Cache<IDType, CachedItemType>::getItemByID(IDType &itemID) {
    typename std::unordered_map<IDType, typename std::list<CachedItem<CachedItemType>>::iterator>::iterator
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

template<typename IDType, typename CachedItemType>
void Cache<IDType, CachedItemType>::flush() {
    // Perhaps not the most efficient, but this method will not be called often anyway
    while(!randomAccessMap.empty()) {
        ejectLeastRecentlyUsedItem();
    }
}

template<typename IDType, typename CachedItemType>
void Cache<IDType, CachedItemType>::touchItem(IDType &itemID) {
    // Move the desired node to the front of the LRU queue
    lruItemQueue.splice(lruItemQueue.begin(), lruItemQueue, randomAccessMap[itemID]);
}

template<typename IDType, typename CachedItemType>
void Cache<IDType, CachedItemType>::insertItem(IDType &itemID, CachedItemType* item) {
    CachedItem<CachedItemType> cachedItem = {false, nullptr};
    cachedItem.item = item;

    // If cache is full, eject a node before we insert the new one
    if(lruItemQueue.size() == itemCapacity) {
        ejectLeastRecentlyUsedItem();
    }

    // When the node is inserted, it is by definition the most recently used one
    // We therefore put it in the front of the queue right away
    lruItemQueue.emplace_front(cachedItem);
    randomAccessMap[itemID] = lruItemQueue.begin();
}

template<typename IDType, typename CachedItemType>
void Cache<IDType, CachedItemType>::markItemDirty(IDType &itemID) {
    typename std::unordered_map<IDType, typename std::list<CachedItem<CachedItemType>>::iterator>::iterator it = randomAccessMap.find(itemID);
    it->second->isDirty = true;
}

// The cache ends up mostly ejecting nodes. Reuse is not all that common.
// We therefore select the node that has been the least recently used, eject it,
// along with all other nodes that will end up in the same dump file.
// This will cause some often used nodes to be ejected, but should mostly reduce the
// number of times a single file is rewritten, thereby improving outflow speed of the cache.
template<typename IDType, typename CachedItemType>
void Cache<IDType, CachedItemType>::ejectLeastRecentlyUsedItem() {
    CachedItem<CachedItemType> leastRecentlyUsedItem = lruItemQueue.back();

    if(leastRecentlyUsedItem.isDirty) {
        eject(leastRecentlyUsedItem.item);

        lruItemQueue.erase(leastRecentlyUsedItem);
        randomAccessMap.erase(leastRecentlyUsedItem);
    }

    delete leastRecentlyUsedItem.item;
}

template<typename IDType, typename CachedItemType>
const CachedItemType *Cache<IDType, CachedItemType>::fetch(IDType &itemID) {
    return getItemByID(itemID);
}