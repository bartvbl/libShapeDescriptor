#include "IndexFileCache.h"

void ejectIndexNode(IndexNodeID nodeID) {

}

void ejectBucketNode(IndexNodeID nodeID) {

}

IndexNode* getIndexNode(IndexNodeID id) {

}

BucketNode* getBucketNode(IndexNodeID id) {

}

const IndexNode *IndexFileCache::fetchIndexNode(IndexNodeID indexNodeID) {
    return nullptr;
}

const BucketNode *IndexFileCache::fetchBucketNode(IndexNodeID bucketNodeID) {
    return nullptr;
}

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

}




