#include <cassert>
#include "IndexFileCache.h"

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

IndexNode *IndexFileCache::getIndexNode(IndexNodeID indexNodeID) {
    std::unordered_map<IndexNodeID, std::list<CachedIndexNode>::iterator>::iterator it = indexNodeMap.find(indexNodeID);
    IndexNode* node = nullptr;

    if(it != indexNodeMap.end())
    {
        // Cache hit
        node = it->second->node;
        touchIndexNode(indexNodeID);
    } else {
        // Cache miss. Read from disk instead
        node = index::io::readIndexNode(indexRoot, indexNodeID, fileGroupSize);
        insertIndexNode(indexNodeID, node);
    }

    return node;
}

BucketNode *IndexFileCache::getBucketNode(IndexNodeID bucketNodeID) {
    std::unordered_map<IndexNodeID, std::list<CachedBucketNode>::iterator>::iterator it = bucketNodeMap.find(bucketNodeID);
    BucketNode* node = nullptr;

    if(it != bucketNodeMap.end())
    {
        // Cache hit
        node = it->second->node;
        touchBucketNode(bucketNodeID);
    } else {
        // Cache miss
        node = index::io::readBucketNode(indexRoot, bucketNodeID, fileGroupSize);
        insertBucketNode(bucketNodeID, node);
    }

    return node;
}

const IndexNode *IndexFileCache::fetchIndexNode(IndexNodeID indexNodeID) {
    return getIndexNode(indexNodeID);
}

const BucketNode *IndexFileCache::fetchBucketNode(IndexNodeID bucketNodeID) {
    return getBucketNode(bucketNodeID);
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
    //std::cout << lruIndexNodeQueue.size() << std::endl;
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

// The cache ends up mostly ejecting nodes. Reuse is not all that common.
// We therefore select the node that has been the least recently used, eject it,
// along with all other nodes that will end up in the same dump file.
// This will cause some often used nodes to be ejected, but should mostly reduce the
// number of times a single file is rewritten, thereby improving outflow speed of the cache.
void IndexFileCache::ejectLeastRecentlyUsedIndexNode() {
    CachedIndexNode leastRecentlyUsedNode = lruIndexNodeQueue.back();

    if(leastRecentlyUsedNode.isDirty) {
        std::vector<IndexNode*> nodesToBeRemoved;
        nodesToBeRemoved.reserve(128);
        const IndexNodeID baseIndex = (leastRecentlyUsedNode.node->id / fileGroupSize) * fileGroupSize;
        for(IndexNodeID i = baseIndex; i < baseIndex + fileGroupSize; i++) {
            std::unordered_map<IndexNodeID, std::list<CachedIndexNode>::iterator>::iterator it = indexNodeMap.find(i);
            if(it != indexNodeMap.end() && it->second->isDirty) {
                nodesToBeRemoved.push_back(it->second->node);
                lruIndexNodeQueue.erase(it->second);
                indexNodeMap.erase(it->second->node->id);
            }
        }

        index::io::writeIndexNodes(indexRoot, nodesToBeRemoved, fileGroupSize);

        for(IndexNode* ejectedNode : nodesToBeRemoved) {
            delete ejectedNode;
        }
    }

}

void IndexFileCache::ejectLeastRecentlyUsedBucketNode() {
    CachedBucketNode leastRecentlyUsedNode = lruBucketNodeQueue.back();

    if(leastRecentlyUsedNode.isDirty) {
        std::vector<BucketNode*> nodesToBeRemoved;
        nodesToBeRemoved.reserve(128);
        const IndexNodeID baseIndex = (leastRecentlyUsedNode.node->id / fileGroupSize) * fileGroupSize;
        for(IndexNodeID i = baseIndex; i < baseIndex + fileGroupSize; i++) {
            std::unordered_map<IndexNodeID, std::list<CachedBucketNode>::iterator>::iterator it = bucketNodeMap.find(i);
            if(it != bucketNodeMap.end() && it->second->isDirty) {
                nodesToBeRemoved.push_back(it->second->node);
                lruBucketNodeQueue.erase(it->second);
                bucketNodeMap.erase(it->second->node->id);
            }
        }

        index::io::writeBucketNodes(indexRoot, nodesToBeRemoved, fileGroupSize);

        for(BucketNode* ejectedNode : nodesToBeRemoved) {
            delete ejectedNode;
        }
    }
}

void IndexFileCache::markIndexNodeDirty(IndexNodeID indexNodeID) {
    std::unordered_map<IndexNodeID, std::list<CachedIndexNode>::iterator>::iterator it = indexNodeMap.find(indexNodeID);
    it->second->isDirty = true;
}

void IndexFileCache::markBucketNodeDirty(IndexNodeID bucketNodeID) {
    std::unordered_map<IndexNodeID, std::list<CachedBucketNode>::iterator>::iterator it = bucketNodeMap.find(bucketNodeID);
    it->second->isDirty = true;
}







