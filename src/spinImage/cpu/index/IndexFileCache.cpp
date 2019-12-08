#include "IndexFileCache.h"

IndexNodeID IndexFileCache::createBucketNode(IndexNodeID parent, unsigned int level) {
    return 0;
}

IndexNodeID IndexFileCache::createIndexNode(IndexNodeID parent, unsigned int level) {
    return 0;
}

IndexNodeID IndexFileCache::promoteBucketNodeToIndexNode(IndexNodeID node) {
    return 0;
}

void IndexFileCache::insertImageIntoBucketNode(IndexNodeID bucketNodeID, IndexEntry entry, unsigned int level) {

}

void IndexFileCache::flush() {
    // Perhaps not the most efficient, but this method will not be called often anyway,
    // and it's better to keep functionality in one place design wise.
    
    while(!indexNodeMap.empty()) {
        ejectLeastRecentlyUsedIndexNode();
    }

    while(!bucketNodeMap.empty()) {
        ejectLeastRecentlyUsedBucketNode();
    }
}

const IndexNode *IndexFileCache::fetchIndexNode(IndexNodeID indexNodeID) {
    std::unordered_map<IndexNodeID, std::list<CachedIndexNode>::iterator>::iterator it = indexNodeMap.find(indexNodeID);
    IndexNode* node = nullptr;

    if(it != indexNodeMap.end())
    {
        // Cache hit
        node = it->second->node;
        touchIndexNode(indexNodeID);
    } else {
        // Cache miss. Read from disk instead
        node = index::io::readIndexNode(indexRoot, indexNodeID);
        insertIndexNode(indexNodeID, node);
    }

    return node;
}

const BucketNode *IndexFileCache::fetchBucketNode(IndexNodeID bucketNodeID) {
    std::unordered_map<IndexNodeID, std::list<CachedBucketNode>::iterator>::iterator it = bucketNodeMap.find(bucketNodeID);
    BucketNode* node = nullptr;

    if(it != bucketNodeMap.end())
    {
        // Cache hit
        node = it->second->node;
        touchBucketNode(bucketNodeID);
    } else {
        // Cache miss
        node = index::io::readBucketNode(indexRoot, bucketNodeID);
        insertBucketNode(bucketNodeID, node);
    }

    return node;
}

void IndexFileCache::touchIndexNode(IndexNodeID indexNodeID) {
    // Move the desired node to the front of the LRU queue
    lruIndexNodeQueue.splice(lruIndexNodeQueue.begin(), lruIndexNodeQueue, indexNodeMap[indexNodeID]);
}

void IndexFileCache::touchBucketNode(IndexNodeID indexNodeID) {
    lruBucketNodeQueue.splice(lruBucketNodeQueue.begin(), lruBucketNodeQueue, bucketNodeMap[indexNodeID]);
}

void IndexFileCache::insertIndexNode(IndexNodeID indexNodeID, IndexNode *node) {
    CachedIndexNode cachedNode = {false, nullptr};
    cachedNode.node = node;

    // Cache is full. Eject a node before we insert the new one
    if(lruIndexNodeQueue.size() == indexNodeCapacity) {
        ejectLeastRecentlyUsedIndexNode();
    }

    // When the node is inserted, it is by definition the most recently used one
    // We therefore put it in the front of the queue right away
    lruIndexNodeQueue.emplace_front(cachedNode);
    indexNodeMap[indexNodeID] = lruIndexNodeQueue.begin();
}

void IndexFileCache::insertBucketNode(IndexNodeID bucketNodeID, BucketNode *node) {
    CachedBucketNode cachedNode = {false, nullptr};
    cachedNode.node = node;

    if(lruBucketNodeQueue.size() == bucketNodeCapacity) {
        ejectLeastRecentlyUsedBucketNode();
    }

    lruBucketNodeQueue.emplace_front(cachedNode);
    bucketNodeMap[bucketNodeID] = lruBucketNodeQueue.begin();
}

void IndexFileCache::ejectLeastRecentlyUsedIndexNode() {
    CachedIndexNode nodeToBeRemoved = lruIndexNodeQueue.back();
    indexNodeMap.erase(nodeToBeRemoved.node->id);
    lruIndexNodeQueue.pop_back();

    if(nodeToBeRemoved.isDirty) {
        index::io::writeIndexNode(indexRoot, nodeToBeRemoved.node);
    }

    delete nodeToBeRemoved.node;
}

void IndexFileCache::ejectLeastRecentlyUsedBucketNode() {
    CachedBucketNode nodeToBeRemoved = lruBucketNodeQueue.back();
    bucketNodeMap.erase(nodeToBeRemoved.node->id);
    lruBucketNodeQueue.pop_back();

    if(nodeToBeRemoved.isDirty) {
        index::io::writeBucketNode(indexRoot, nodeToBeRemoved.node);
    }

    delete nodeToBeRemoved.node;
}






