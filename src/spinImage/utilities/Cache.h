#pragma once


#include <list>
#include <unordered_map>
#include <cassert>

// The cached nodes are stored as pointers to avoid accidental copies being created
template<typename IDType, typename CachedItemType> struct CachedItem {
    bool isDirty;
    IDType ID;
    CachedItemType* item;
};

template<typename IDType, typename CachedItemType> class Cache {
private:
    // Nodes are evicted on a Least Recently Used basis
    // This is most efficiently done by using a doubly linked list
    std::list<CachedItem<IDType, CachedItemType>> lruItemQueue;

    // These hash tables allow efficient fetching of nodes from the cache
    std::unordered_map<IDType, typename std::list<CachedItem<IDType, CachedItemType>>::iterator> randomAccessMap;


protected:

    // Get hold of an item. May cause another item to be ejected
    CachedItemType* getItemByID(IDType &itemID) {
        typename std::unordered_map<IDType, typename std::list<CachedItem<IDType, CachedItemType>>::iterator>::iterator
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

    // Insert an item into the cache. May cause another item to be ejected
    void insertItem(IDType &itemID, CachedItemType* item) {
        CachedItem<IDType, CachedItemType> cachedItem = {false, "", nullptr};
        cachedItem.ID = itemID;
        cachedItem.item = item;

        // If cache is full, eject a node before we insert the new one
        if (lruItemQueue.size() == itemCapacity) {
            ejectLeastRecentlyUsedItem();
        }

        // When the node is inserted, it is by definition the most recently used one
        // We therefore put it in the front of the queue right away
        lruItemQueue.emplace_front(cachedItem);
        randomAccessMap[itemID] = lruItemQueue.begin();
    }

    // Mark an item present in the cache as most recently used
    void touchItem(IDType &itemID) {
        // Move the desired node to the front of the LRU queue
        lruItemQueue.splice(lruItemQueue.begin(), lruItemQueue, randomAccessMap[itemID]);
    }

    // Set the dirty flag of a given item
    void markItemDirty(IDType &itemID) {
        typename std::unordered_map<IDType, typename std::list<CachedItem<IDType, CachedItemType>>::iterator>::iterator it = randomAccessMap.find(itemID);
        assert(it != randomAccessMap.end());
        it->second->isDirty = true;
    }

    void ejectLeastRecentlyUsedItem() {
        CachedItem<IDType, CachedItemType> leastRecentlyUsedItem = lruItemQueue.back();

        if(leastRecentlyUsedItem.isDirty) {
            eject(leastRecentlyUsedItem.item);

            typename std::list<CachedItem<IDType, CachedItemType>>::iterator it_start =
                    randomAccessMap.find(leastRecentlyUsedItem.ID)->second;
            this->lruItemQueue.erase(it_start);
            this->randomAccessMap.erase(leastRecentlyUsedItem.ID);
        }

        delete leastRecentlyUsedItem.item;
    }

    // What needs to happen when a cache miss or eviction occurs depends on the specific use case
    // Since this class is a general implementation, a subclass needs to implement this functionality.
    virtual void eject(CachedItemType* item) = 0;
    virtual CachedItemType* load(IDType &itemID) = 0;

    explicit Cache(const size_t capacity) : itemCapacity(capacity) {
        // std::list does not have a reserve()
        //lruItemQueue.reserve(capacity);
        randomAccessMap.reserve(capacity);
    }
public:
    // The lookup functions returns const pointers to ensure the only copy of these item exist in the cache
    // It also ensures the cache handles any necessary changes, as nodes need to be marked as dirty
    const CachedItemType* fetch(IDType &itemID) {
        return getItemByID(itemID);
    }

    // Eject all items from the cache, leave it empty
    void flush() {
        while(!lruItemQueue.empty()) {
            ejectLeastRecentlyUsedItem();
        }
    }

    size_t getCurrentItemCount() {
        return lruItemQueue.size();
    }

    const size_t itemCapacity;
};