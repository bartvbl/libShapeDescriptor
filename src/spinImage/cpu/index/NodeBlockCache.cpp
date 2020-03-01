#include <cassert>
#include "NodeBlockCache.h"

void NodeBlockCache::eject(NodeBlock *block) {
    SpinImage::index::io::writeNodeBlock(block, indexRoot);
}

NodeBlock *NodeBlockCache::load(std::string &itemID) {
    return SpinImage::index::io::readNodeBlock(itemID, indexRoot);
}

unsigned char computeLevelByte(const MipmapStack &mipmaps, const unsigned short level) {
    // Level 1 contains 8 1-byte chunks
    if(level < 8) {
        return mipmaps.level1.image >> (64U - 8U * (level + 1)) & 0xFFU;
    }

    // Level 2 starts after level 1, and contains 16 columns of 2 chunks each
    const unsigned short level2StartChunk = 8 + (16 * 2);
    if(level < level2StartChunk) {
        // TODO: Transpose 16x16 image, change bit order within column to be more balanced
        const unsigned short chunkInlevel2 = level - 8;
        const unsigned short level2UintIndex = chunkInlevel2 / 4;
        const unsigned short level2ByteIndex = chunkInlevel2 % 4;
        const unsigned int level2Uint = mipmaps.level2.image[level2UintIndex];
        return level2Uint >> (32U - 8U * (level2ByteIndex + 1)) & 0xFFU;
    }

    // Level 3 starts after level 2, and contains 32 columns of 4 chunks each
    if(level < level2StartChunk + (32 * 4)) {
        const unsigned short chunkInlevel3 = level - 8;
        const unsigned short level3UintIndex = chunkInlevel3 / 4;
        const unsigned short level3ByteIndex = chunkInlevel3 % 4;
        const unsigned int level3Uint = mipmaps.level3.image[level3UintIndex];
        return level3Uint >> (32U - 8U * (level3ByteIndex + 1)) & 0xFFU;
    }

    // Index has run out of intermediate levels, and this function should never be called in that case.
    throw std::runtime_error("Level byte requested from mipmap stack that is out of bounds!");
}



void NodeBlockCache::insertImageIntoNode(const QuiccImage &image, const IndexEntry &entry, NodeBlock *currentNodeBlock,
                          unsigned char outgoingEdgeIndex) {
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
    // Create and insert new node into cache
    NodeBlock* childNodeBlock = new NodeBlock();
    childNodeBlock->identifier = childNodeID;
    insertItem(childNodeID, childNodeBlock);
    markItemDirty(childNodeID);

    std::cout << "s" << std::flush;
    //std::cout << "Splitting into new node " << childNodeID << " (" << getCurrentItemCount() << "/" << itemCapacity << ")" << std::endl;

    // Follow linked list and move all nodes into new child node block
    int nextLinkedNodeIndex = currentNodeBlock->leafNodeContentsStartIndices.at(outgoingEdgeIndex);
    while(nextLinkedNodeIndex != -1) {
        // Copy over node into new child node block
        NodeBlockEntry* entryToMove = &currentNodeBlock->leafNodeContents.at(nextLinkedNodeIndex);
        MipmapStack entryMipmaps(entryToMove->image);
        unsigned char childLevelByte = computeLevelByte(entryMipmaps, levelReached + 1);
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

// It's a waste to recreate this one every time, so let's reuse it
std::stringstream pathBuilder;

void NodeBlockCache::insertImage(const QuiccImage &image, const IndexEntry reference) {
    // Follow path until leaf node is reached, or the bottom of the index
    unsigned short levelReached = 0;
    // Clear the path/identifier buffer
    pathBuilder.str("");
    pathBuilder << std::hex;

    bool currentNodeIsLeafNode = false;
    NodeBlock* currentNodeBlock = rootNode;
    MipmapStack mipmaps(image);
    while(!currentNodeIsLeafNode) {
        unsigned char outgoingEdgeIndex = computeLevelByte(mipmaps, levelReached);
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
                levelReached < 8 + (2 * 16) + (4 * 32)) {
                pathBuilder << (outgoingEdgeIndex < 16 ? "0" : "") << int(outgoingEdgeIndex) << "/";
                std::string childNodeID = pathBuilder.str();
                splitNode(levelReached, currentNodeBlock, outgoingEdgeIndex, childNodeID);
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




