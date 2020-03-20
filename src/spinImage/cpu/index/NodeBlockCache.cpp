#include <cassert>
#include <spinImage/cpu/index/types/BitCountMipmapStack.h>
#include "NodeBlockCache.h"

size_t countImages(std::array<std::vector<NodeBlockEntry>, NODES_PER_BLOCK> &entries) {
    unsigned int entryCount = 0;
    for(const auto& entry : entries) {
        entryCount += entry.size();
    }
    return entryCount;
}

// May be called by multiple threads simultaneously
void NodeBlockCache::eject(NodeBlock *block) {
    #pragma omp atomic
    nodeBlockStatistics.totalWriteCount++;
    auto writeStart = std::chrono::high_resolution_clock::now();

    SpinImage::index::io::writeNodeBlock(block, indexRoot);

    auto writeEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::nano> writeDuration = writeEnd - writeStart;
    #pragma omp atomic
    nodeBlockStatistics.totalWriteTimeNanoseconds += writeDuration.count();
}

// May be called by multiple threads simultaneously
NodeBlock *NodeBlockCache::load(std::string &itemID) {
    #pragma omp atomic
    nodeBlockStatistics.totalReadCount++;
    auto readStart = std::chrono::high_resolution_clock::now();

    NodeBlock* readBlock = SpinImage::index::io::readNodeBlock(itemID, indexRoot);

    auto readEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::nano> readDuration = readEnd - readStart;
    #pragma omp atomic
    nodeBlockStatistics.totalReadTimeNanoseconds += readDuration.count();
    #pragma omp atomic
    currentImageCount += countImages(readBlock->leafNodeContents);

    return readBlock;
}

void NodeBlockCache::onEviction(NodeBlock *block) {
    #pragma omp atomic
    currentImageCount -= countImages(block->leafNodeContents);
}

bool shouldSplit(unsigned int leafNodeSize, unsigned int levelReached, bool isBottomLevel) {
    return !isBottomLevel ?
          leafNodeSize >= NODE_SPLIT_THRESHOLD
        : leafNodeSize >= NODE_SPLIT_THRESHOLD * 64;
}

std::string shortToHex(unsigned long byte) {
    std::string byteString;
    const std::string characterMap = "0123456789abcdef";
    byteString += characterMap.at((byte >> 12U) & 0x0FU);
    byteString += characterMap.at((byte >> 8U) & 0x0FU);
    byteString += characterMap.at((byte >> 4U) & 0x0FU);
    byteString += characterMap.at((byte & 0x0FU));
    return byteString;
}
void NodeBlockCache::splitNode(
        unsigned short levelReached,
        NodeBlock *currentNodeBlock,
        unsigned long outgoingEdgeIndex,
        IndexPath &indexPath,
        std::string &childNodeID) {
    #pragma omp atomic
    nodeBlockStatistics.nodeSplitCount++;

    assert(currentNodeBlock->childNodeIsLeafNode[outgoingEdgeIndex]);

    // Create and insert new node into cache
    NodeBlock* childNodeBlock = new NodeBlock();
    childNodeBlock->blockLock.lock();
    childNodeBlock->identifier = childNodeID;

    if(!indexPath.isBottomLevel(levelReached)) {
        // Follow linked list and move all nodes into new child node block
        for(const auto& entryToMove : currentNodeBlock->leafNodeContents.at(outgoingEdgeIndex))
        {
            BitCountMipmapStack entryMipmapStack(entryToMove.image);
            IndexPath entryGuidePath(entryMipmapStack);
            childNodeBlock->leafNodeContents.at(entryGuidePath.at(levelReached + 1)).push_back(entryToMove);
        }

        // If any node in the new child block is full, that one needs to be split as well
        for(unsigned int childIndex = 0; childIndex < NODES_PER_BLOCK; childIndex++) {
            if(shouldSplit(childNodeBlock->leafNodeContents.at(childIndex).size(), levelReached + 1, indexPath.isBottomLevel(levelReached + 1))) {
                std::string splitNodeID = childNodeID + shortToHex(childIndex) + "/";
                splitNode(levelReached + 1, childNodeBlock, childIndex, indexPath, splitNodeID);
            }
        }

        // Clear memory occupied by child node
        std::vector<NodeBlockEntry>().swap(currentNodeBlock->leafNodeContents.at(outgoingEdgeIndex));
    }

    // Mark the entry in the node block as an intermediate node
    currentNodeBlock->childNodeIsLeafNode.set(outgoingEdgeIndex, false);

    // Add item to the cache
    insertItem(childNodeID, childNodeBlock, true);
    childNodeBlock->blockLock.unlock();
}

void NodeBlockCache::insertImage(const QuiccImage &image, const IndexEntry reference) {
    #pragma omp atomic
    nodeBlockStatistics.imageInsertionCount++;
    #pragma omp atomic
    currentImageCount++;

    // Follow path until leaf node is reached, or the bottom of the index
    unsigned short levelReached = 0;
    // Clear the path/identifier buffer
    std::stringstream pathBuilder;
    pathBuilder << std::hex;

    bool currentNodeIsLeafNode = false;
    std::string currentNodeID;
    // currentNodeID initialises to "", which causes this to fetch the root node
    NodeBlock* currentNodeBlock = borrowItemByID(currentNodeID);
    currentNodeBlock->blockLock.lock();
    BitCountMipmapStack mipmaps(image);
    IndexPath indexPath = IndexPath(mipmaps);
    while(!currentNodeIsLeafNode) {
        unsigned long outgoingEdgeIndex = indexPath.at(levelReached);
        if(currentNodeBlock->childNodeIsLeafNode[outgoingEdgeIndex] == true) {
            // Leaf node reached. Insert image into it
            currentNodeIsLeafNode = true;
            std::string itemID = pathBuilder.str();
            currentNodeBlock->leafNodeContents.at(outgoingEdgeIndex).push_back(NodeBlockEntry(reference, image));

            // 2. Mark modified entry as dirty.
            // Do this first to avoid cases where item is going to ejected from the cache when node is split
            markItemDirty(itemID);

            // 3. Split if threshold has been reached, but not if we're at the deepest possible level
            if(shouldSplit(currentNodeBlock->leafNodeContents.at(outgoingEdgeIndex).size(), levelReached, indexPath.isBottomLevel(levelReached))) {
                pathBuilder << shortToHex(outgoingEdgeIndex) << "/";
                std::string childNodeID = pathBuilder.str();

                auto splitStart = std::chrono::high_resolution_clock::now();

                splitNode(levelReached, currentNodeBlock, outgoingEdgeIndex, indexPath, childNodeID);

                auto splitEnd = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::nano> splitDuration = splitEnd - splitStart;
                #pragma omp atomic
                nodeBlockStatistics.totalSplitTimeNanoseconds += splitDuration.count();
            }
            currentNodeBlock->blockLock.unlock();
            returnItemByID(currentNodeID);
        } else {
            currentNodeBlock->blockLock.unlock();
            returnItemByID(currentNodeID);
            // Fetch child of intermediate node, then start the process over again.
            levelReached++;
            pathBuilder << shortToHex(outgoingEdgeIndex) << "/";
            currentNodeID = pathBuilder.str();
            assert(indexPath.isBottomLevel(levelReached) || currentNodeBlock->leafNodeContents.at(outgoingEdgeIndex).empty());
            currentNodeBlock = borrowItemByID(currentNodeID);
            assert(currentNodeID == currentNodeBlock->identifier);
            currentNodeBlock->blockLock.lock();
        }
    }

    // Clean up the cache to ensure memory usage doesn't get out of hand
    while(currentImageCount > imageCapacity) {
        forceLeastRecentlyUsedEviction();
    }
}

const NodeBlock* NodeBlockCache::getNodeBlockByID(std::string blockID) {
    NodeBlock* block = borrowItemByID(blockID);
    returnItemByID(blockID);
    return block;
}

size_t NodeBlockCache::getCurrentImageCount() const {
    return currentImageCount;
}




