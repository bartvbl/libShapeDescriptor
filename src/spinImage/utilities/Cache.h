#pragma once

#include <mutex>
#include <list>
#include <unordered_map>
#include <cassert>
#include <omp.h>
#include <iostream>
#include <condition_variable>
#include <thread>
#include <vector>
#include <malloc.h>

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

    std::mutex queueLock;
    std::condition_variable queueConditionVariable;

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
        assert(!evictedItem.isInUse);
        assert(randomAccessMap.find(evictedItem.ID) != randomAccessMap.end());
        leastRecentlyUsedItem->isInUse = true;


        typename std::list<CachedItem<IDType, CachedItemType>>::iterator it = std::next(leastRecentlyUsedItem).base();

        if(evictedItem.isDirty) {
            statistics.dirtyEvictions++;

            cacheLock.unlock();
            eject(evictedItem.item);
            cacheLock.lock();
        }

        onEviction(evictedItem.item);


        assert(it->ID == evictedItem.ID);
        this->lruItemQueue.erase(it);
        this->randomAccessMap.erase(evictedItem.ID);
        assert(randomAccessMap.find(evictedItem.ID) == randomAccessMap.end());

        delete evictedItem.item;
    }

    CacheLookupResult<CachedItemType> attemptItemLookup(IDType &itemID) {
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
        cacheLock.unlock();
        return {true, cachedItemEntry};
    }

    void doItemInsertion(IDType &itemID, CachedItemType* item, bool dirty = false, bool borrow = false) {
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
        cacheLock.lock();
        evictLeastRecentlyUsedItem();
        cacheLock.unlock();
    }

protected:
    // Get hold of an item. May cause another item to be ejected. Marks item as in use.
    CachedItemType* borrowItemByID(IDType &itemID) {
        std::unique_lock<std::mutex> mainLock(queueLock);
        CacheLookupResult<CachedItemType> lookupResult = {false, nullptr};
        unsigned int failedCount = 0;
        while(!lookupResult.lookupSuccessful) {
            lookupResult.lookupSuccessful = false;
            // THREAD UNSAFE LOOKUP!!
            // Done to ensure only threads don't hold up other threads trying to do useful stuff in the cache
            typename std::unordered_map<IDType, typename std::list<CachedItem<IDType, CachedItemType>>::iterator>::iterator
                    it = randomAccessMap.find(itemID);
            // If the item is not in the cache, it can't be contested, so we can go ahead and grab it.
            // If the item is in the cache, take a quick peek to see if it is available right now
            if((it == randomAccessMap.end() || (it != randomAccessMap.end() && !it->second->isInUse)) || failedCount >= 100) {
                lookupResult = attemptItemLookup(itemID);
                failedCount = 0;
            } else {
                failedCount++;
            }
            if(!lookupResult.lookupSuccessful) {
                queueConditionVariable.wait_until(mainLock, std::chrono::steady_clock::now() + std::chrono::nanoseconds(10000));
            }
        }
        // Wake up next thread to go in
        queueConditionVariable.notify_all();
        return lookupResult.item;
    }

    void returnItemByID(IDType &itemID) {
        cacheLock.lock();
        typename std::unordered_map<IDType, typename std::list<CachedItem<IDType, CachedItemType>>::iterator>::iterator
                it = randomAccessMap.find(itemID);
        assert(it != randomAccessMap.end());
        assert(it->second->isInUse);
        it->second->isInUse = false;
        cacheLock.unlock();
    }

    // Insert an item into the cache. May cause another item to be ejected. Marks item as in use.
    void insertItem(IDType &itemID, CachedItemType* item, bool dirty = false, bool borrowItem = false) {
        cacheLock.lock();
        doItemInsertion(itemID, item, dirty, borrowItem);
        cacheLock.unlock();
    }

    // Set the dirty flag of a given item.
    void markItemDirty(IDType &itemID) {
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
        cacheLock.lock();

        size_t flushedCount = 0;
        #pragma omp parallel
        #pragma omp single
        {
            for (auto item = lruItemQueue.begin(); item != lruItemQueue.end(); ++item) {
                #pragma omp task firstprivate(item)
                {
                    #pragma omp atomic
                    statistics.evictions++;
                    if(item->isDirty) {
                        #pragma omp atomic
                        statistics.dirtyEvictions++;
                        eject(item->item);
                    }
                    onEviction(item->item);
                    delete item->item;

                    #pragma omp atomic
                    flushedCount++;
                    std::cout << "\rRemaining: " + std::to_string(lruItemQueue.size() - flushedCount) + "     " << std::flush;

                    if (flushedCount % 1000 == 0) {
                        malloc_trim(0);
                    }
                }
            }
            #pragma omp taskwait
        };



        lruItemQueue.clear();
        randomAccessMap.clear();
        cacheLock.unlock();
        assert(lruItemQueue.empty());
        assert(randomAccessMap.empty());
    }

    // NOT thread safe!
    size_t getCurrentItemCount() const {
        return lruItemQueue.size();
    }
};