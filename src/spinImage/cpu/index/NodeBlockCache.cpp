#include <cassert>
#include "NodeBlockCache.h"

void NodeBlockCache::eject(NodeBlock *block) {
    index::io::writeNodeBlock(block, indexRoot);
}

NodeBlock *NodeBlockCache::load(std::string &itemID) {
    return index::io::loadNodeBlock(itemID, indexRoot);
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

void NodeBlockCache::insertImageIntoNode(const MipMapLevel3 &mipmaps, const IndexEntry &entry, NodeBlock *currentNodeBlock,
                          unsigned char levelByte, std::string &itemID) {
    // 1. Insert the new entry at the start of the list
    int currentStartIndex = currentNodeBlock->entryStartIndices.at(levelByte);
    int entryIndex = currentNodeBlock->entries.size();
    currentNodeBlock->entries.push_back({entry, mipmaps, currentStartIndex});
    currentNodeBlock->entryStartIndices.at(levelByte) = entryIndex;

    // 2. Mark modified entry as dirty.
    // Do this first to avoid cases where item is going to ejected from the cache when node is split
    markItemDirty(itemID);
}

// It's a waste to recreate this one every time, so let's reuse it
std::stringstream pathBuilder;

void NodeBlockCache::insertImage(const MipmapStack &mipmaps, const IndexEntry reference) {
    // Follow path until leaf node is reached, or the bottom of the index
    unsigned short levelReached = 0;
    // Clear the path/identifier buffer
    pathBuilder.str(std::string());

    bool currentNodeIsLeafNode = false;
    NodeBlock* currentNodeBlock = rootNode;
    while(!currentNodeIsLeafNode) {
        unsigned char levelByte = computeLevelByte(mipmaps, levelReached);
        pathBuilder << levelByte;
        if(currentNodeBlock->childNodeIsLeafNode[levelByte] == true) {
            // Leaf node reached. Insert image into it
            currentNodeIsLeafNode = true;
            std::string itemID = pathBuilder.str();
            insertImageIntoNode(mipmaps, reference, currentNodeBlock, levelByte, itemID);

            // 3. Split if threshold has been reached
            if(currentNodeBlock->entries.size() >= NODE_SPLIT_THRESHOLD) {
                for(int child = 0; child < NODES_PER_BLOCK; child++) {
                    // Skip past intermediate nodes
                    if(currentNodeBlock->childNodeIsLeafNode[child] == false) {
                        assert(currentNodeBlock->entryStartIndices.at(child) == -1);
                        continue;
                    }

                    // If the leaf node has no contents, it does not need to be split
                    if(currentNodeBlock->entryStartIndices.at(child) == -1) {
                        continue;
                    }

                    // The current entry is a leaf node and has contents.
                    // We therefore create a new node block, and divide its contents over it.
                    NodeBlock* childNodeBlock = new NodeBlock();

                    unsigned char childLevelByte = computeLevelByte(mipmaps, levelReached + 1);
                    std::string childItemID = itemID + "/" + std::to_string(childLevelByte);
                    insertItem(childItemID, childNodeBlock);
                    markItemDirty(childItemID);

                    // Follow linked list and move all nodes into new child node block
                    int nextLinkedNodeIndex = currentNodeBlock->entryStartIndices.at(child);
                    while(nextLinkedNodeIndex != -1) {
                        NodeBlockEntry entryToMove = currentNodeBlock->entries.at(nextLinkedNodeIndex);
                        insertImageIntoNode(entryToMove.mipmapImage, entryToMove.indexEntry,
                                childNodeBlock, childLevelByte, childItemID);

                        nextLinkedNodeIndex = entryToMove.nextEntryIndex;
                    }

                    // Mark the entry in the node block as an intermediate node
                    currentNodeBlock->entryStartIndices.at(child) = -1;
                    currentNodeBlock->childNodeIsLeafNode[child] = false;
                }

                currentNodeBlock->entries.clear();
            }

        } else {
            // Fetch child of intermediateNode, then start the process over again.
            levelReached++;
            pathBuilder << "/";
            std::string nextNodeID = pathBuilder.str();
            currentNodeBlock = getItemByID(nextNodeID);
        }
    }
}


