#include <cassert>
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

void NodeBlockCache::splitNode(
        unsigned short levelReached,
        NodeBlock *currentNodeBlock,
        unsigned char outgoingEdgeIndex,
        std::string &childNodeID) {
    #pragma omp atomic
    nodeBlockStatistics.nodeSplitCount++;

    //std::cout << "s" << std::flush;

    // Create and insert new node into cache
    NodeBlock* childNodeBlock = new NodeBlock();
    childNodeBlock->blockLock.lock();
    childNodeBlock->identifier = childNodeID;

    // Follow linked list and move all nodes into new child node block
    for(const auto& entryToMove : currentNodeBlock->leafNodeContents.at(outgoingEdgeIndex))
    {
        // Copy over node into new child node block
        MipmapStack entryMipmaps(entryToMove.image);
        // Look at the next byte in the mipmap to determine which child bucket will receive the child node
        unsigned char childLevelByte = entryMipmaps.computeLevelByte(levelReached + 1);
        childNodeBlock->leafNodeContents.at(childLevelByte).push_back(entryToMove);
    }

    // Mark the entry in the node block as an intermediate node
    std::vector<NodeBlockEntry>().swap(currentNodeBlock->leafNodeContents.at(outgoingEdgeIndex));
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
    MipmapStack mipmaps(image);
    while(!currentNodeIsLeafNode) {
        unsigned char outgoingEdgeIndex = mipmaps.computeLevelByte(levelReached);
        if(currentNodeBlock->childNodeIsLeafNode[outgoingEdgeIndex] == true) {
            // Leaf node reached. Insert image into it
            currentNodeIsLeafNode = true;
            std::string itemID = pathBuilder.str();
            currentNodeBlock->leafNodeContents.at(outgoingEdgeIndex).push_back(NodeBlockEntry(reference, image));

            // 2. Mark modified entry as dirty.
            // Do this first to avoid cases where item is going to ejected from the cache when node is split
            markItemDirty(itemID);

            // 3. Split if threshold has been reached, but not if we're at the deepest possible level
            if(currentNodeBlock->leafNodeContents.at(outgoingEdgeIndex).size() >= NODE_SPLIT_THRESHOLD &&
                    (levelReached < 8 + (2 * 16) + (4 * 32) - 1)) {
                pathBuilder << (outgoingEdgeIndex < 16 ? "0" : "") << int(outgoingEdgeIndex) << "/";
                std::string childNodeID = pathBuilder.str();

                auto splitStart = std::chrono::high_resolution_clock::now();

                splitNode(levelReached, currentNodeBlock, outgoingEdgeIndex, childNodeID);

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
            // Fetch child of intermediateNode, then start the process over again.
            levelReached++;
            pathBuilder << (outgoingEdgeIndex < 16 ? "0" : "") << int(outgoingEdgeIndex) << "/";
            currentNodeID = pathBuilder.str();
            assert(currentNodeBlock->leafNodeContents.at(outgoingEdgeIndex).empty());
            currentNodeBlock = borrowItemByID(currentNodeID);
            currentNodeBlock->blockLock.lock();
        }
    }

    // Clean up the cache to ensure memory usage doesn't get out of hand
    while(currentImageCount > imageCapacity) {
        forceLeastRecentlyUsedEviction();
    }
}

size_t NodeBlockCache::getCurrentImageCount() const {
    return currentImageCount;
}




