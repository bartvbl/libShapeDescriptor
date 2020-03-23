#pragma once

#include <experimental/filesystem>
#include <utility>
#include <list>
#include <unordered_map>
#include <spinImage/cpu/types/QuiccImage.h>
#include <spinImage/utilities/Cache.h>
#include <spinImage/cpu/index/types/IndexPath.h>
#include <set>
#include "IndexIO.h"

/* Notes on where I'm at right now
 * - Each node block represents the links going out of a node
 * - Splitting a node involves marking the parent as an intermediate node, and creating a new node block. Since each node block can hold quite a few images, we shouldn't end up in a case where we have an insane number of dangling nodes with only a few images in them. Also, nodes have the flexibility of having specific nodes marked as intermediate ones, so branches are covered that way, and where more space is needed they are expanded
 * - One file per node block for easy implementation. Each byte "jump" is also a folder, such that folders don't get insane numbers of nodes in them
 * - LZMA2 was the best of the tested algorithms that yielded the best compression ratios
 * - Each node block shares a std::vector to save on allocations and memory overhead. Each element inside this vector contains an index entry, a mipmap image, and an integer that references the next index at which an image can be found. For each leaf node, only the first index is stored in the list, which still saves memory over having a separate vector for each node. Of course, as the number of images this grows, the memory usage is more expensive, but I somewhat consider this an investment in speed, as fewer memory allocations need to be done this way.
 * - The aforementioned linked list should probably be rearranged prior to dumping it into a file, for better cache utilisation when iterating over images and splitting them. Shouldn't be too expensive either.
 * - The index tree must at all times be fully connected. So when an intermediate node is split, a new leaf node must be created for all 256 links. This saves 32 bytes per intermediate node to keep track of a particular link is populated. Since memory is managed in node blocks anyway, trying to make things more granular only adds to the complexity of the overall implementation.
 * - Lookups are done by keeping an open/closed priority queue much like A*. You start with the root node, which you remove from the queue, then you look at its links, and sort them by their distance function rating with respect to the image being queried. When you encounter a leaf node, add its contents into the priority queue too. Stop when no more intermediate nodes exist within the first X entries. This lookup can also ignore pixels that are not relevant to the image being queried, much like the clutter resistance property.
 * - File format is already implemented, just write something in IndexIO that serialises a node block. Next you simply compress the whole thing at once.
 * - 
 */

struct NodeBlockCacheStatistics {
    size_t imageInsertionCount = 0;
    size_t nodeSplitCount = 0;
    size_t totalReadCount = 0;
    size_t totalWriteCount = 0;
    double totalReadTimeNanoseconds = 0;
    double totalWriteTimeNanoseconds = 0;
    double totalSplitTimeNanoseconds = 0;

    void reset() {
        imageInsertionCount = 0;
        nodeSplitCount = 0;
        totalReadCount = 0;
        totalWriteCount = 0;
        totalReadTimeNanoseconds = 0;
        totalWriteTimeNanoseconds = 0;
        totalSplitTimeNanoseconds = 0;
    }
};

class NodeBlockCache : public Cache<std::string, NodeBlock> {
private:
    const std::experimental::filesystem::path indexRoot;
    size_t currentImageCount = 0;

    void splitNode(
            unsigned short levelReached,
            NodeBlock *currentNodeBlock,
            unsigned long levelByte,
            IndexPath &indexPath,
            std::string &childNodeID);
protected:
    void eject(NodeBlock* item) override;
    void onEviction(NodeBlock* item) override;
    NodeBlock* load(std::string &itemID) override;
public:
    NodeBlockCacheStatistics nodeBlockStatistics;
    const size_t imageCapacity;

    NodeBlockCache(
            size_t nodeBlockCapacity,
            size_t imageCapacity,
            const std::experimental::filesystem::path &indexRootPath,
            bool loadExisting = false)
    :   Cache(nodeBlockCapacity),
        indexRoot(indexRootPath),
        imageCapacity(imageCapacity)
        {
            if(!loadExisting) {
                std::string rootNodeID;
                insertItem(rootNodeID, new NodeBlock(), true);
            }
        }
    void insertImage(const QuiccImage &image, const IndexEntry reference);
    const NodeBlock* getNodeBlockByID(std::string blockID);
    size_t getCurrentImageCount() const;
};