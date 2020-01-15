#include <cassert>
#include "IndexFileCache.h"


/*
IndexNodeID IndexFileCache::createLink(const IndexNodeID parent, const unsigned int* mipmapImage, const unsigned int parentLevel, const unsigned int LINK_TYPE) {
    IndexNodeID createdNodeID = nextNodeID;
    nextNodeID++;

    IntermediateNode* parentNode = fetch(parent);
    const unsigned int arraySizes[4] = {0, 2, 8, 32};
    unsigned int imageArrayLength = arraySizes[parentLevel];

    parentNode->images.insert(parentNode->images.end(), mipmapImage, mipmapImage + imageArrayLength);
    parentNode->linkTypes.push_back(LINK_TYPE);
    parentNode->links.emplace_back(createdNodeID);


    return createdNodeID;
}

IndexNodeID BucketNodeCache::createBucketNode(const IndexNodeID parent, const unsigned int* mipmapImage, const unsigned int parentLevel) {
    IndexNodeID newBucketNodeID = createLink(parent, mipmapImage, parentLevel, 0xFFFFFFFFU);
    BucketNode* bucketNode = new BucketNode(newBucketNodeID);
    insertItem(newBucketNodeID, bucketNode);
    markItemDirty(newBucketNodeID);

    return newBucketNodeID;
}

IndexNodeID IndexNodeCache::createIndexNode(const IndexNodeID parent, const unsigned int *mipmapImage, const unsigned int parentLevel) {
    IndexNodeID newIndexNodeID = createLink(parent, mipmapImage, parentLevel, 0xFFFFFFFFU);
    IntermediateNode* indexNode = new IntermediateNode(newIndexNodeID);
    insertItem(newIndexNodeID, indexNode);
    markItemDirty(newIndexNodeID);

    return newIndexNodeID;
}


void BucketNodeCache::insertImageIntoBucketNode(IndexNodeID bucketNodeID, IndexEntry entry) {
    BucketNode* bucketNode = getItemByID(bucketNodeID);
    bucketNode->images.emplace_back(entry);
    markItemDirty(bucketNodeID);
    touchItem(bucketNodeID);
}
 */