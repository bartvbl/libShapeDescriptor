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

// It's a waste to recreate this one every time, so let's reuse it
std::stringstream pathBuilder;

void NodeBlockCache::insertImage(const MipmapStack &mipmaps, const IndexEntry reference) {
    // Follow path until leaf node is reached, or the bottom of the index
    unsigned short levelReached = 0;
    // Clear the path/identifier buffer
    pathBuilder.str(std::string());

    bool currentNodeIsLeafNode = false;
    NodeBlock* currentIntermediateNode = rootNode;
    while(!currentNodeIsLeafNode) {
        unsigned char levelByte = computeLevelByte(mipmaps, levelReached);
        pathBuilder << levelByte;
        if(currentIntermediateNode->nodeExists[levelByte] == false) {
            // End of tree has been reached, but leaf node is missing.
            // Create leaf node, insert image into it
            currentNodeIsLeafNode = true;
        } else if(currentIntermediateNode->linkTypes[levelByte]) {
            // Leaf node reached. Insert image into it
            currentNodeIsLeafNode = true;
            //insertImageIntoLeafNode();
        } else {
            // Node link must exist, and the next node is again an intermediate node.
            // Fetch it, then start the process over again.
            levelReached++;
            pathBuilder << "/";
            std::string nextNodeID = pathBuilder.str();
            currentIntermediateNode = getItemByID(nextNodeID);
        }
    }
}
