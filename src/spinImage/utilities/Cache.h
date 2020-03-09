#pragma once

#include <mutex>
#include <list>
#include <unordered_map>
#include <cassert>
#include <omp.h>
#include <iostream>
#include <condition_variable>
#include <thread>

// The cached nodes are stored as pointers to avoid accidental copies being created
template<typename IDType, typename CachedItemType> struct CachedItem {
    bool isDirty;
    bool isInUse;
    IDType ID;
    CachedItemType* item;
};

template<typename CachedItemType> struct CacheLookupResult {
    bool lookupSuccessful;
    CachedItemType* item;
};

struct CacheStatistics {
    size_t misses = 0;
    size_t hits = 0;
    size_t evictions = 0;
    size_t dirtyEvictions = 0;
    size_t insertions = 0;

    void reset() {
        misses = 0;
        hits = 0;
        evictions = 0;
        dirtyEvictions = 0;
        insertions = 0;
    }
};

template<typename IDType, typename CachedItemType> class Cache {
public:
    // Nodes are evicted on a Least Recently Used basis
    // This is most efficiently done by using a doubly linked list
    std::list<CachedItem<IDType, CachedItemType>> lruItemQueue;

    // These hash tables allow efficient fetching of nodes from the cache
    std::unordered_map<IDType, typename std::list<CachedItem<IDType, CachedItemType>>::iterator> randomAccessMap;

    // Lock used for modification of the cache data structures.
    std::mutex cacheLock;

    // List which keeps track of which entries are in the process of being loaded/unloaded
    std::vector<IDType> beingLoadedList;

    // Mark an item present in the cache as most recently used
    void touchItem(IDType &itemID) {
        // Move the desired node to the front of the LRU queue
        typename std::unordered_map<IDType, typename std::list<CachedItem<IDType, CachedItemType>>::iterator>::iterator
                it = randomAccessMap.find(itemID);
        assert(it != randomAccessMap.end());
        assert(it->second->ID == itemID);
        lruItemQueue.splice(lruItemQueue.begin(), lruItemQueue, it->second);
    }

    unsigned int indexOfItemBeingLoaded(IDType &id) {
        for(unsigned int i = 0; i < beingLoadedList.size(); i++) {
            if(beingLoadedList[i] == id) {
                return i;
            }
        }
        return beingLoadedList.size();
    }

    void evictLeastRecentlyUsedItem() {
        statistics.evictions++;
        typename std::list<CachedItem<IDType, CachedItemType>>::reverse_iterator leastRecentlyUsedItem;

        bool foundEntry = false;
        for(auto entry = lruItemQueue.rbegin(); entry != lruItemQueue.rend(); ++entry) {
            if(!entry->isInUse) {
                leastRecentlyUsedItem = entry;
                foundEntry = true;
                break;
            }
        }

        if(!foundEntry) {
            return;
        }

        // Make a copy so we don't rely on the iterator
        CachedItem<IDType, CachedItemType> evictedItem = *leastRecentlyUsedItem;
        //std::cout << "Thread " + std::to_string(omp_get_thread_num()) + " is ejecting item " + evictedItem.ID + "\n" << std::flush;
        assert(!evictedItem.isInUse);
        assert(randomAccessMap.find(evictedItem.ID) != randomAccessMap.end());
        leastRecentlyUsedItem->isInUse = true;

        onEviction(evictedItem.item);

        typename std::list<CachedItem<IDType, CachedItemType>>::iterator it = std::next(leastRecentlyUsedItem).base();

        if(evictedItem.isDirty) {
            statistics.dirtyEvictions++;

            cacheLock.unlock();
            eject(evictedItem.item);
            cacheLock.lock();
        }

        assert(it->ID == evictedItem.ID);
        this->lruItemQueue.erase(it);
        this->randomAccessMap.erase(evictedItem.ID);
        assert(randomAccessMap.find(evictedItem.ID) == randomAccessMap.end());

        delete evictedItem.item;
    }

    CacheLookupResult<CachedItemType> attemptItemLookup(IDType &itemID) {
        //std::cout << "Thread " + std::to_string(omp_get_thread_num()) + " is attempting lookup of item " + itemID + "\n" << std::flush;
        cacheLock.lock();
        typename std::unordered_map<IDType, typename std::list<CachedItem<IDType, CachedItemType>>::iterator>::iterator
                it = randomAccessMap.find(itemID);
        CachedItemType* cachedItemEntry = nullptr;

        if(it != randomAccessMap.end())
        {
            assert(it->second->ID == itemID);

            // Cache hit
            if(it->second->isInUse) {
                // Collision!
                // The thread which is trying to read this value will have to come back later
                //std::cout << "Thread " + std::to_string(omp_get_thread_num()) + " has experienced a collision!\n" << std::flush;
                cacheLock.unlock();
                return {false, nullptr};
            } else {
                statistics.hits++;
                cachedItemEntry = it->second->item;
                it->second->isInUse = true;
                touchItem(itemID);
            }
        } else {
            // Cache miss. Load the item into the cache instead
            bool isBeingLoaded = indexOfItemBeingLoaded(itemID) != beingLoadedList.size();
            if(isBeingLoaded) {
                // The entry is already in the process of being loaded into memory by another thread
                // We therefore abort this lookup and come back later
                cacheLock.unlock();
                return {false, nullptr};
            }
            //std::cout << "Thread " + std::to_string(omp_get_thread_num()) + " has experienced a cache miss on item " + itemID + " and is loading it from disk!\n" << std::flush;
            statistics.misses++;
            beingLoadedList.push_back(itemID);
            cacheLock.unlock();
            cachedItemEntry = load(itemID);
            cacheLock.lock();
            doItemInsertion(itemID, cachedItemEntry);
            it = randomAccessMap.find(itemID);
            it->second->isInUse = true;
            beingLoadedList.erase(beingLoadedList.begin() + indexOfItemBeingLoaded(itemID));
        }
        //std::cout << "Thread " + std::to_string(omp_get_thread_num()) + " is borrowing " + itemID + "\n" << std::flush;
        cacheLock.unlock();
        return {true, cachedItemEntry};
    }

    void doItemInsertion(IDType &itemID, CachedItemType* item, bool dirty = false, bool borrow = false) {
        //std::cout << "Thread " + std::to_string(omp_get_thread_num()) + " is inserting a new item with ID " + itemID + "\n" << std::flush;

        CachedItem<IDType, CachedItemType> cachedItem = {false, false, "", nullptr};
        cachedItem.ID = itemID;
        cachedItem.item = item;
        cachedItem.isDirty = dirty;
        cachedItem.isInUse = borrow;

        // If cache is full (or due to a race condition over capacity),
        // make space before we insert the new one
        while(lruItemQueue.size() >= itemCapacity) {
            evictLeastRecentlyUsedItem();
        }

        // When the node is inserted, it is by definition the most recently used one
        // We therefore put it in the front of the queue right away
        lruItemQueue.emplace_front(cachedItem);
        randomAccessMap[itemID] = lruItemQueue.begin();

        statistics.insertions++;
    }

    void forceLeastRecentlyUsedEviction() {
        //std::cout << "Thread " + std::to_string(omp_get_thread_num()) + " is forcing an eviction\n" << std::flush;
        cacheLock.lock();
        evictLeastRecentlyUsedItem();
        cacheLock.unlock();
    }

protected:
    // Get hold of an item. May cause another item to be ejected. Marks item as in use.
    CachedItemType* borrowItemByID(IDType &itemID) {
        //std::cout << "Thread " + std::to_string(omp_get_thread_num()) + " wants to borrow item " + itemID + "\n" << std::flush;
        CacheLookupResult<CachedItemType> lookupResult = {false, nullptr};
        while(!lookupResult.lookupSuccessful) {
            lookupResult = attemptItemLookup(itemID);
            if(!lookupResult.lookupSuccessful) {
                //std::cout << "Thread " + std::to_string(omp_get_thread_num()) + " has determined item " + itemID + " is currently in use and is waiting\n" << std::flush;
                std::this_thread::sleep_for (std::chrono::milliseconds(10));
            }
        }
        //std::cout << "Thread " + std::to_string(omp_get_thread_num()) + " is now borrowing item " + itemID + "\n" << std::flush;
        return lookupResult.item;
    }

    void returnItemByID(IDType &itemID) {
        //std::cout << "Thread " + std::to_string(omp_get_thread_num()) + " is returning item " + itemID + "\n" << std::flush;
        cacheLock.lock();
        typename std::unordered_map<IDType, typename std::list<CachedItem<IDType, CachedItemType>>::iterator>::iterator
                it = randomAccessMap.find(itemID);
        assert(it != randomAccessMap.end());
        assert(it->second->isInUse);
        it->second->isInUse = false;
        cacheLock.unlock();
        //std::cout << "Thread " + std::to_string(omp_get_thread_num()) + " has now returned item " + itemID + "\n" << std::flush;
    }

    // Insert an item into the cache. May cause another item to be ejected. Marks item as in use.
    void insertItem(IDType &itemID, CachedItemType* item, bool dirty = false, bool borrowItem = false) {
        //std::cout << "Thread " + std::to_string(omp_get_thread_num()) + " is inserting item " + itemID + "\n" << std::flush;
        cacheLock.lock();
        doItemInsertion(itemID, item, dirty, borrowItem);
        cacheLock.unlock();
    }

    // Set the dirty flag of a given item.
    void markItemDirty(IDType &itemID) {
        //std::cout << "Thread " + std::to_string(omp_get_thread_num()) + " is marking item " + itemID + " as dirty\n" << std::flush;
        cacheLock.lock();
        typename std::unordered_map<IDType, typename std::list<CachedItem<IDType, CachedItemType>>::iterator>::iterator it = randomAccessMap.find(itemID);
        assert(it != randomAccessMap.end());
        // Not a thorough check that everything is as it should be,
        // but the item should at the very least be loaned out.
        // Ideally we'd also check for whether the owner of the item is the thread calling this function
        // but this is a decent sanity check.
        assert(it->second->isInUse);
        it->second->isDirty = true;
        cacheLock.unlock();
    }

    // What needs to happen when a cache miss or eviction occurs depends on the specific use case
    // Since this class is a general implementation, a subclass needs to implement this functionality.

    // May be called by multiple threads simultaneously
    virtual void onEviction(CachedItemType* item) = 0;
    // May be called by multiple threads simultaneously
    virtual void eject(CachedItemType* item) = 0;
    // May be called by multiple threads simultaneously
    virtual CachedItemType* load(IDType &itemID) = 0;

    explicit Cache(const size_t capacity) : itemCapacity(capacity) {
        randomAccessMap.reserve(capacity);
    }
public:
    const size_t itemCapacity;
    CacheStatistics statistics;

    // Eject all items from the cache, leave it empty
    void flush() {
#pragma omp parallel
        {
            while(lruItemQueue.size() > 0) {
                std::cout << std::to_string(lruItemQueue.size()) + "\n";
                forceLeastRecentlyUsedEviction();
            }
        };
    }

    // NOT thread safe!
    size_t getCurrentItemCount() const {
        return lruItemQueue.size();
    }


};