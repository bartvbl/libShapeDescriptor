#pragma once


#include <list>
#include <unordered_map>

// The cached nodes are stored as pointers to avoid accidental copies being created
template<typename type> struct CachedItem {
    bool isDirty;
    type* item;
};

template<typename IDType, typename CachedItemType> class Cache {
private:
    // Nodes are evicted on a Least Recently Used basis
    // This is most efficiently done by using a doubly linked list
    std::list<CachedItem<CachedItemType>> lruItemQueue;

    // These hash tables allow efficient fetching of nodes from the cache
    std::unordered_map<IDType, typename std::list<CachedItem<CachedItemType>>::iterator> randomAccessMap;

    const size_t itemCapacity;
protected:
    // Get hold of an item. May cause another item to be ejected
    CachedItemType* getItemByID(IDType &itemID);

    // Insert an item into the cache. May cause another item to be ejected
    void insertItem(IDType &itemID, CachedItemType* item);

    // Mark an item present in the cache as most recently used
    void touchItem(IDType &itemID);

    // Set the dirty flag of a given item
    void markItemDirty(IDType &itemID);

    void ejectLeastRecentlyUsedItem();

    // What needs to happen when a cache miss or eviction occurs depends on the specific use case
    // Since this class is a general implementation, a subclass needs to implement this functionality.
    virtual void eject(CachedItemType* item) = 0;
    virtual CachedItemType* load(IDType &itemID) = 0;

    explicit Cache(const size_t capacity) : itemCapacity(capacity) {
        lruItemQueue.resize(capacity);
        randomAccessMap.reserve(capacity);
    }
public:
    // The lookup functions returns const pointers to ensure the only copy of these item exist in the cache
    // It also ensures the cache handles any necessary changes, as nodes need to be marked as dirty
    const CachedItemType* fetch(IDType &itemID);

    // Eject all items from the cache, leave it empty
    void flush();
};