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

bool shouldSplit(unsigned int leafNodeSize, bool isBottomLevel) {
    return !isBottomLevel ?
          leafNodeSize >= NODE_SPLIT_THRESHOLD
        : leafNodeSize >= NODE_SPLIT_THRESHOLD * 64;
}

void NodeBlockCache::splitNode(
        NodeBlock *currentNodeBlock,
        unsigned long outgoingEdgeIndex,
        IndexPath &pathInIndex) {
    #pragma omp atomic
    nodeBlockStatistics.nodeSplitCount++;

    assert(currentNodeBlock->childNodeIsLeafNode[outgoingEdgeIndex]);
    IndexPath childPathInIndex = pathInIndex;
    childPathInIndex.append(outgoingEdgeIndex);
    std::string childNodeID = childPathInIndex.to_string();

    // Create and insert new node into cache
    NodeBlock* childNodeBlock = new NodeBlock();
    childNodeBlock->blockLock.lock();
    childNodeBlock->identifier = childNodeID;

    if(!childPathInIndex.isBottomLevel()) {
        // Follow linked list and move all nodes into new child node block
        for(const auto& entryToMove : *currentNodeBlock->getNodeContentsByIndex(outgoingEdgeIndex))
        {
            // Look at the next byte in the mipmap to determine which child bucket will receive the child node
            BitCountMipmapStack entryMipmapStack(entryToMove.image);
            IndexPath entryGuidePath(entryMipmapStack);
            unsigned long childNodeDirection = entryGuidePath.at(pathInIndex.length() + 1);
            childNodeBlock->getNodeContentsByIndex(childNodeDirection)->push_back(entryToMove);
        }

        // If any node in the new child block is full, that one needs to be split as well
        for(unsigned long childOutgoingEdgeIndex : *childNodeBlock->getOutgoingEdgeIndices()) {
            if(shouldSplit(childNodeBlock->getNodeContentsByIndex(childOutgoingEdgeIndex)->size(), childPathInIndex.isBottomLevel())) {
                splitNode(childNodeBlock, childOutgoingEdgeIndex, childPathInIndex);
            }
        }

        // Clear memory occupied by child node
        std::vector<NodeBlockEntry>().swap(*currentNodeBlock->getEntriesByOutgoingEdgeIndex(outgoingEdgeIndex));
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

    bool currentNodeIsLeafNode = false;

    // currentNodeID initialises to "", which causes this to fetch the root node
    std::string currentNodeID;
    NodeBlock* currentNodeBlock = borrowItemByID(currentNodeID);

    currentNodeBlock->blockLock.lock();
    BitCountMipmapStack mipmaps(image);
    IndexPath guidePath = IndexPath(mipmaps);
    IndexPath pathInIndex = IndexPath();

    while(!currentNodeIsLeafNode) {
        unsigned long outgoingEdgeIndex = guidePath.at(pathInIndex.length());
        if(currentNodeBlock->childNodeIsLeafNode[outgoingEdgeIndex] == true) {
            // Leaf node reached. Insert image into it
            currentNodeIsLeafNode = true;
            currentNodeBlock->insert(outgoingEdgeIndex, NodeBlockEntry(reference, image));

            // 2. Mark modified entry as dirty.
            // Do this first to avoid cases where item is going to ejected from the cache when node is split
            std::string itemID = pathInIndex.to_string();
            markItemDirty(itemID);

            // 3. Split if threshold has been reached, but not if we're at the deepest possible level
            if(shouldSplit(currentNodeBlock->getNodeContentsByIndex(outgoingEdgeIndex)->size(), pathInIndex.isBottomLevel())) {

                auto splitStart = std::chrono::high_resolution_clock::now();

                splitNode(currentNodeBlock, outgoingEdgeIndex, pathInIndex);

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
            pathInIndex.append(guidePath.at(pathInIndex.length()));
            currentNodeID = pathInIndex.to_string();
            assert(pathInIndex.isBottomLevel() || currentNodeBlock->leafNodeContents.at(outgoingEdgeIndex).empty());
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




