#pragma once

#include <mutex>
#include <list>
#include <unordered_map>
#include <cassert>
#include <omp.h>
#include <iostream>
#include <condition_variable>

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
private:
    // Nodes are evicted on a Least Recently Used basis
    // This is most efficiently done by using a doubly linked list
    std::list<CachedItem<IDType, CachedItemType>> lruItemQueue;

    // These hash tables allow efficient fetching of nodes from the cache
    std::unordered_map<IDType, typename std::list<CachedItem<IDType, CachedItemType>>::iterator> randomAccessMap;

    // Lock used for modification of the cache data structures.
    // Reentrant property is necessary for
    std::mutex cacheLock;

    // Condition variable for sleeping threads until
    std::mutex waitMutex;
    std::condition_variable waitCV;

    // Mark an item present in the cache as most recently used
    void touchItem(IDType &itemID) {
        // Move the desired node to the front of the LRU queue
        typename std::unordered_map<IDType, typename std::list<CachedItem<IDType, CachedItemType>>::iterator>::iterator
                it = randomAccessMap.find(itemID);
        bool doesContain = it != randomAccessMap.end();
        assert(doesContain);
        auto itemReference = randomAccessMap[itemID];
        lruItemQueue.splice(lruItemQueue.begin(), lruItemQueue, itemReference);
    }

    void evictLeastRecentlyUsedItem() {
        statistics.evictions++;
        CachedItem<IDType, CachedItemType>* leastRecentlyUsedItem = nullptr;

        for(auto entry = lruItemQueue.rbegin(); entry != lruItemQueue.rend(); ++entry) {
            if(!(*entry).isInUse) {
                leastRecentlyUsedItem = &(*entry);
                break;
            }
        }

        if(leastRecentlyUsedItem->isDirty) {
            statistics.dirtyEvictions++;
            leastRecentlyUsedItem->isInUse = true;

            cacheLock.unlock();
            eject(leastRecentlyUsedItem->item);
            cacheLock.lock();
        }

        typename std::list<CachedItem<IDType, CachedItemType>>::iterator it_start =
                randomAccessMap.find(leastRecentlyUsedItem->ID)->second;
        this->lruItemQueue.erase(it_start);
        this->randomAccessMap.erase(leastRecentlyUsedItem->ID);

        delete leastRecentlyUsedItem->item;
    }

    CacheLookupResult<CachedItemType> attemptItemLookup(IDType &itemID) {
        cacheLock.lock();
        typename std::unordered_map<IDType, typename std::list<CachedItem<IDType, CachedItemType>>::iterator>::iterator
                it = randomAccessMap.find(itemID);
        CachedItemType* cachedItemEntry = nullptr;

        if(it != randomAccessMap.end())
        {
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
            statistics.misses++;
            // TODO: this should be done in parallel, but requires some way of marking that the item is already being loaded
            cachedItemEntry = load(itemID);
            insertItem(itemID, cachedItemEntry);
        }
        cacheLock.unlock();
        return {true, cachedItemEntry};
    }

protected:
    // Get hold of an item. May cause another item to be ejected. Marks item as in use.
    CachedItemType* borrowItemByID(IDType &itemID) {
        CacheLookupResult<CachedItemType> lookupResult = {false, nullptr};
        while(!lookupResult.lookupSuccessful) {
            lookupResult = attemptItemLookup(itemID);
            if(!lookupResult.lookupSuccessful) {
                //std::cout << "Thread " + std::to_string(omp_get_thread_num()) + " has determined item " + itemID + " is currently in use and is waiting\n" << std::flush;
                {
                    std::unique_lock<std::mutex> lock(waitMutex);
                    waitCV.wait(lock);
                }
            }
        }
        //std::cout << "Thread " + std::to_string(omp_get_thread_num()) + " is now borrowing item " + itemID + "\n" << std::flush;
        return lookupResult.item;
    }

    void returnItemByID(IDType &itemID) {
        cacheLock.lock();
        typename std::unordered_map<IDType, typename std::list<CachedItem<IDType, CachedItemType>>::iterator>::iterator
                it = randomAccessMap.find(itemID);
        assert(it != randomAccessMap.end());
        assert(it->second->isInUse);
        it->second->isInUse = false;
        {
            std::unique_lock<std::mutex> lock(waitMutex);
            waitCV.notify_all();
        }
        //std::cout << "Thread " + std::to_string(omp_get_thread_num()) + " has now returned item " + itemID + "\n" << std::flush;
        cacheLock.unlock();
    }

    // Insert an item into the cache. May cause another item to be ejected. Marks item as in use.
    void insertItem(IDType &itemID, CachedItemType* item, bool dirty = false) {
        cacheLock.lock();
        //std::cout << "Thread " + std::to_string(omp_get_thread_num()) + " is inserting a new item with ID " + itemID + "\n" << std::flush;

        statistics.insertions++;
        CachedItem<IDType, CachedItemType> cachedItem = {false, false, "", nullptr};
        cachedItem.ID = itemID;
        cachedItem.item = item;
        cachedItem.isDirty = dirty;

        // If cache is full (or due to a race condition over capacity),
        // make space before we insert the new one
        while(lruItemQueue.size() >= itemCapacity) {
            evictLeastRecentlyUsedItem();
        }

        // When the node is inserted, it is by definition the most recently used one
        // We therefore put it in the front of the queue right away
        lruItemQueue.emplace_front(cachedItem);
        randomAccessMap[itemID] = lruItemQueue.begin();

        cacheLock.unlock();
    }

    // Set the dirty flag of a given item.
    void markItemDirty(IDType &itemID) {
        cacheLock.lock();
        //std::cout << "Thread " + std::to_string(omp_get_thread_num()) + " is marking item " + itemID + " as dirty\n" << std::flush;
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
    virtual void eject(CachedItemType* item) = 0;
    // May be called by multiple threads simultaneously
    virtual CachedItemType* load(IDType &itemID) = 0;

    explicit Cache(const size_t capacity) : itemCapacity(capacity) {
        // std::list does not have a reserve()
        //lruItemQueue.reserve(capacity);
        randomAccessMap.reserve(capacity);
    }
public:
    const size_t itemCapacity;
    CacheStatistics statistics;

    // Eject all items from the cache, leave it empty
    void flush() {
#pragma omp parallel
        {
            size_t index = 0;
            for(auto item = lruItemQueue.begin(); item != lruItemQueue.end(); item++) {
                if(index % omp_get_num_threads() != omp_get_thread_num()) {
                    index++;
                    continue;
                }

                if((*item).isDirty) {
                    eject((*item).item);
                }

                delete (*item).item;

                index++;
            }
        };

        // Empty all cached items
        lruItemQueue.clear();
        randomAccessMap.clear();
    }

    size_t getCurrentItemCount() const {
        return lruItemQueue.size();
    }


};