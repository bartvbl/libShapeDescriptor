#include <cassert>
#include "NodeBlockCache.h"

void NodeBlockCache::eject(NodeBlock *block) {
    nodeBlockStatistics.totalWriteCount++;
    auto writeStart = std::chrono::high_resolution_clock::now();

    SpinImage::index::io::writeNodeBlock(block, indexRoot);

    auto writeEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::nano> writeDuration = writeEnd - writeStart;
    nodeBlockStatistics.totalWriteTimeNanoseconds += writeDuration.count();
}

NodeBlock *NodeBlockCache::load(std::string &itemID) {
    nodeBlockStatistics.totalReadCount++;
    auto readStart = std::chrono::high_resolution_clock::now();

    NodeBlock* readBlock = SpinImage::index::io::readNodeBlock(itemID, indexRoot);

    auto readEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::nano> readDuration = readEnd - readStart;
    nodeBlockStatistics.totalReadTimeNanoseconds += readDuration.count();

    return readBlock;
}

void NodeBlockCache::insertImageIntoNode(const QuiccImage &image, const IndexEntry &entry, NodeBlock *currentNodeBlock, unsigned char outgoingEdgeIndex) {
    // 1. Insert the new entry at the start of the list
    int currentStartIndex = currentNodeBlock->leafNodeContentsStartIndices.at(outgoingEdgeIndex);
    int entryIndex = -1;
    if(currentNodeBlock->freeListStartIndex != -1) {
        // We have spare capacity from nodes that were deleted previously
        entryIndex = currentNodeBlock->freeListStartIndex;
        int nextFreeListIndex = currentNodeBlock->leafNodeContents.at(entryIndex).nextEntryIndex;
        currentNodeBlock->leafNodeContents.at(entryIndex) = {entry, image, currentStartIndex};
        currentNodeBlock->freeListStartIndex = nextFreeListIndex;
    } else {
        // Otherwise, when no spare capacity is available,
        // we allocate a new node at the end of the list
        entryIndex = currentNodeBlock->leafNodeContents.size();
        currentNodeBlock->leafNodeContents.push_back({entry, image, currentStartIndex});
    }
    currentNodeBlock->leafNodeContentsStartIndices.at(outgoingEdgeIndex) = entryIndex;
    currentNodeBlock->leafNodeContentsLength.at(outgoingEdgeIndex)++;
}

void NodeBlockCache::splitNode(
        unsigned short levelReached,
        NodeBlock *currentNodeBlock,
        unsigned char outgoingEdgeIndex,
        std::string &childNodeID) {
    nodeBlockStatistics.nodeSplitCount++;

    // Create and insert new node into cache
    NodeBlock* childNodeBlock = new NodeBlock();
    childNodeBlock->identifier = childNodeID;
    insertItem(childNodeID, childNodeBlock);
    markItemDirty(childNodeID);

    // Follow linked list and move all nodes into new child node block
    int nextLinkedNodeIndex = currentNodeBlock->leafNodeContentsStartIndices.at(outgoingEdgeIndex);
    while(nextLinkedNodeIndex != -1) {
        // Copy over node into new child node block
        NodeBlockEntry* entryToMove = &currentNodeBlock->leafNodeContents.at(nextLinkedNodeIndex);
        MipmapStack entryMipmaps(entryToMove->image);
        unsigned char childLevelByte = entryMipmaps.computeLevelByte(levelReached + 1);
        insertImageIntoNode(entryToMove->image, entryToMove->indexEntry,
                            childNodeBlock, childLevelByte);

        // Mark entry in parent node as available by appending it to the free list
        int nextFreeEntryIndex = currentNodeBlock->freeListStartIndex;
        entryToMove->nextEntryIndex = nextFreeEntryIndex;
        currentNodeBlock->freeListStartIndex = nextLinkedNodeIndex;

        // Moving on to next entry in the linked list
        nextLinkedNodeIndex = entryToMove->nextEntryIndex;
    }

    // Mark the entry in the node block as an intermediate node
    currentNodeBlock->leafNodeContentsStartIndices.at(outgoingEdgeIndex) = -1;
    currentNodeBlock->leafNodeContentsLength.at(outgoingEdgeIndex) = 0;
    currentNodeBlock->childNodeIsLeafNode.set(outgoingEdgeIndex, false);
}

void NodeBlockCache::insertImage(const QuiccImage &image, const IndexEntry reference) {
    nodeBlockStatistics.imageInsertionCount++;

    // Follow path until leaf node is reached, or the bottom of the index
    unsigned short levelReached = 0;
    // Clear the path/identifier buffer
    std::stringstream pathBuilder;
    pathBuilder << std::hex;

    bool currentNodeIsLeafNode = false;
    NodeBlock* currentNodeBlock = rootNode;
    MipmapStack mipmaps(image);
    while(!currentNodeIsLeafNode) {
        unsigned char outgoingEdgeIndex = mipmaps.computeLevelByte(levelReached);
        if(currentNodeBlock->childNodeIsLeafNode[outgoingEdgeIndex] == true) {
            // Leaf node reached. Insert image into it
            currentNodeIsLeafNode = true;
            std::string itemID = pathBuilder.str();
            insertImageIntoNode(image, reference, currentNodeBlock, outgoingEdgeIndex);

            // 2. Mark modified entry as dirty.
            // Do this first to avoid cases where item is going to ejected from the cache when node is split
            // Root node is never ejected, and not technically part of the cache, so we avoid it
            if(levelReached > 0) {
                markItemDirty(itemID);
            }

            // 3. Split if threshold has been reached, but not if we're at the deepest possible level
            if(currentNodeBlock->leafNodeContentsLength.at(outgoingEdgeIndex) >= NODE_SPLIT_THRESHOLD &&
                    (levelReached < 8 + (2 * 16) + (4 * 32) - 1)) {
                pathBuilder << (outgoingEdgeIndex < 16 ? "0" : "") << int(outgoingEdgeIndex) << "/";
                std::string childNodeID = pathBuilder.str();

                auto splitStart = std::chrono::high_resolution_clock::now();

                splitNode(levelReached, currentNodeBlock, outgoingEdgeIndex, childNodeID);

                auto splitEnd = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::nano> splitDuration = splitEnd - splitStart;
                nodeBlockStatistics.totalSplitTimeNanoseconds += splitDuration.count();
            }

        } else {
            // Fetch child of intermediateNode, then start the process over again.
            levelReached++;
            pathBuilder << (outgoingEdgeIndex < 16 ? "0" : "") << int(outgoingEdgeIndex) << "/";
            std::string nextNodeID = pathBuilder.str();
            currentNodeBlock = getItemByID(nextNodeID);
        }
    }
}




