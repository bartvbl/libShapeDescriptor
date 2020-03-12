#include <queue>
#include "IndexQueryer.h"
#include "NodeBlockCache.h"
#include <spinImage/cpu/index/types/BitCountMipmapStack.h>
#include <algorithm>
#include <climits>

typedef std::array<unsigned char, 8 + (16 * 16) + (32 * 32)> IndexPath;

struct UnvisitedNode {
    UnvisitedNode(IndexPath indexPath, std::string unvisitedNodeID, unsigned int minDistance, unsigned int nodeLevel)
    : path(indexPath), nodeID(unvisitedNodeID), minDistanceScore(minDistance), level(nodeLevel) {}

    IndexPath path;
    std::string nodeID;
    unsigned int minDistanceScore;
    unsigned int hammingDistanceScore;
    unsigned int level;

    // We want the open node priority queue to sort items by lowest score
    // Since the priority queue by default optimises for finding the highest sorted element,
    // we need to invert the sort order.
    bool operator< (const UnvisitedNode &right) const {
        return minDistanceScore > right.minDistanceScore;
    }
};

struct SearchResultEntry {
    SearchResultEntry(IndexEntry entry, const QuiccImage &imageEntry, unsigned int minDistance)
        : reference(entry), image(imageEntry), distanceScore(minDistance) {}

    IndexEntry reference;
    QuiccImage image;
    unsigned int distanceScore;

    bool operator< (const SearchResultEntry &right) const {
        return distanceScore < right.distanceScore;
    }
};

std::stringstream IDBuilder;

std::string appendPath(const std::string &parentNodeID, unsigned char childIndex) {
    std::string byteString = parentNodeID;
    const std::string characterMap = "0123456789abcdef";
    byteString += characterMap.at((childIndex >> 4U) & 0x0FU);
    byteString += characterMap.at((childIndex & 0x0FU));
    byteString += "/";
    return byteString;
}

unsigned int computeDistance(const QuiccImage &needle, const QuiccImage &haystack) {
    unsigned int score = 0;
    for(int i = 0; i < needle.size(); i++) {
        score += std::bitset<32>((needle[i] ^ haystack[i]) & needle[i]).count();
    }
    return score;
}

unsigned int computeLevelByteMinDistance(const std::array<unsigned short, 8> &bitDistances, const unsigned char levelByte) {
    std::bitset<8> levelByteBits(levelByte);
    return levelByteBits[7] * (bitDistances[0] == 0 ? 1 : 0) +
           levelByteBits[6] * (bitDistances[1] == 0 ? 1 : 0) +
           levelByteBits[5] * (bitDistances[2] == 0 ? 1 : 0) +
           levelByteBits[4] * (bitDistances[3] == 0 ? 1 : 0) +
           levelByteBits[3] * (bitDistances[4] == 0 ? 1 : 0) +
           levelByteBits[2] * (bitDistances[5] == 0 ? 1 : 0) +
           levelByteBits[1] * (bitDistances[6] == 0 ? 1 : 0) +
           levelByteBits[0] * (bitDistances[7] == 0 ? 1 : 0);
}

unsigned int computeMinDistance(
        const BitCountMipmapStack &needle,
        const IndexPath &haystack,
        unsigned int upToLevel) {
    std::array<unsigned short, 8> partialScores =
            {0, 0, 0, 0, 0, 0, 0, 0};

    unsigned int columnIndex = 0;
    unsigned short columnMinDistance = 0;
    for(int level = 0; level <= upToLevel; level++) {
        unsigned char levelByte = haystack[level];

        std::array<unsigned short, 8> bitDistances;

        if(level < 8) {
            bitDistances = {
                needle.level1[level * 8 + 0],
                needle.level1[level * 8 + 1],
                needle.level1[level * 8 + 2],
                needle.level1[level * 8 + 3],
                needle.level1[level * 8 + 4],
                needle.level1[level * 8 + 5],
                needle.level1[level * 8 + 6],
                needle.level1[level * 8 + 7]
            };
        } else if(level < 8 + 16) {
            bitDistances = {
                needle.level2[(level - 8) * 8 + 0],
                needle.level2[(level - 8) * 8 + 1],
                needle.level2[(level - 8) * 8 + 2],
                needle.level2[(level - 8) * 8 + 3],
                needle.level2[(level - 8) * 8 + 4],
                needle.level2[(level - 8) * 8 + 5],
                needle.level2[(level - 8) * 8 + 6],
                needle.level2[(level - 8) * 8 + 7]
            };
        } else {
            bitDistances = {
                    needle.level3[(level - 8 - (2 * 16)) * 8 + 0],
                    needle.level3[(level - 8 - (2 * 16)) * 8 + 1],
                    needle.level3[(level - 8 - (2 * 16)) * 8 + 2],
                    needle.level3[(level - 8 - (2 * 16)) * 8 + 3],
                    needle.level3[(level - 8 - (2 * 16)) * 8 + 4],
                    needle.level3[(level - 8 - (2 * 16)) * 8 + 5],
                    needle.level3[(level - 8 - (2 * 16)) * 8 + 6],
                    needle.level3[(level - 8 - (2 * 16)) * 8 + 7]
            };
        }

        // Compute and update partial min distance score
        columnMinDistance += computeLevelByteMinDistance(bitDistances, levelByte);

            // Level 1
        if  ((level < 8) ||
            // Level 2
            ((level >= 8) && (level < 8 + (2 * 16)) && (level % 4 == 3)) ||
            // Level 3
            ((level >= 8 + (2 * 16)) && (level % 16 == 15))) {
            // Scores can only increase, and since we look at parts of columns at a time,
            partialScores[columnIndex] = std::max<unsigned short>(columnMinDistance, partialScores[columnIndex]);
            columnIndex++;
            columnIndex = columnIndex % 8;
            columnMinDistance = 0;
        }
    }
    // Also commit final partial column
    partialScores[columnIndex] = std::max<unsigned short>(columnMinDistance, partialScores[columnIndex]);

    // Finally, compute sum of partial sums
    unsigned int minDistanceSum = 0;
    for(unsigned short partialScore : partialScores) {
        minDistanceSum += partialScore;
    }
    return minDistanceSum;
}

const unsigned int computeMinDistanceThreshold(std::vector<SearchResultEntry> &currentSearchResults) {
    return currentSearchResults.empty() ?
               INT_MAX
               : currentSearchResults.at(currentSearchResults.size() - 1).distanceScore;
}

void visitNode(
        const NodeBlock* block,
        const IndexPath &path,
        const std::string &nodeID,
        const unsigned int level,
        std::priority_queue<UnvisitedNode> &closedNodeQueue,
        std::vector<UnvisitedNode> &debug_closedNodeQueue,
        std::vector<SearchResultEntry> &currentSearchResults,
        const BitCountMipmapStack &queryImageMipmapStack,
        const QuiccImage &queryImage) {
    // Divide child nodes over both queues
    const unsigned int childLevel = level + 1;
    // If we have not yet acquired any search results, disable the threshold
    const unsigned int searchResultScoreThreshold =
            computeMinDistanceThreshold(currentSearchResults);

    std::cout << "Visiting node " << currentSearchResults.size() << " search results, " << closedNodeQueue.size() << " queued nodes, " << searchResultScoreThreshold  << " vs " << closedNodeQueue.top().minDistanceScore << " - " << nodeID << std::endl;
    for(int child = 0; child < NODES_PER_BLOCK; child++) {
        if(block->childNodeIsLeafNode[child]) {
            //std::cout << "Child " << child << " is leaf node!" << std::endl;
            // If child is a leaf node, insert its images into the search result list
            for(const NodeBlockEntry& entry : block->leafNodeContents.at(child)) {
                unsigned int distanceScore = computeDistance(queryImage, entry.image);

                // Only consider the image if it is potentially better than what's there already
                if(distanceScore <= searchResultScoreThreshold) {
                    currentSearchResults.emplace_back(entry.indexEntry, entry.image, distanceScore);
                }
            }
        } else {
            // If the child is an intermediate node, enqueue it in the closed node list
            IndexPath childPath = path;
            childPath[level] = (unsigned char) child;
            unsigned int minDistanceScore = computeMinDistance(queryImageMipmapStack, childPath, level);

            if(minDistanceScore <= searchResultScoreThreshold) {
                //unsigned int hammingDistance = computeHammingDistance(queryImageMipmapStack, childPath, level);
                //std::cout << "Enqueued " << appendPath(nodeID, child) << " -> " << minDistanceScore << std::endl;
                //std::cout << "Enqueued " << appendPath(nodeID, child) << " -> " << minDistanceScore << std::endl;
                closedNodeQueue.emplace(
                    childPath,
                    appendPath(nodeID, child),
                    minDistanceScore,
                    childLevel);
                debug_closedNodeQueue.emplace_back(
                    childPath,
                    appendPath(nodeID, child),
                    minDistanceScore,
                    childLevel);
            }
        }
    }
}

std::vector<SpinImage::index::QueryResult> SpinImage::index::query(Index &index, const QuiccImage &queryImage, unsigned int resultCount) {
    BitCountMipmapStack queryImageBitCountMipmapStack(queryImage);

    NodeBlockCache cache(100000, 2500000, index.indexDirectory, true);

    std::priority_queue<UnvisitedNode> closedNodeQueue;
    std::vector<UnvisitedNode> debug_closedNodeQueue;
    std::vector<SearchResultEntry> currentSearchResults;

    currentSearchResults.reserve(30000 + resultCount + NODES_PER_BLOCK * NODE_SPLIT_THRESHOLD);

    // Root node path is not referenced, so can be left uninitialised
    IndexPath rootNodePath = {0};
    closedNodeQueue.emplace(rootNodePath, "", 0, 0);
    debug_closedNodeQueue.emplace_back(rootNodePath, "", 0, 0);

    // Iteratively add additional nodes until there's no chance any additional node can improve the best distance score
    while(  !closedNodeQueue.empty() &&
            computeMinDistanceThreshold(currentSearchResults) > closedNodeQueue.top().minDistanceScore) {
        UnvisitedNode nextBestUnvisitedNode = closedNodeQueue.top();
        closedNodeQueue.pop();
        debug_closedNodeQueue.erase(debug_closedNodeQueue.begin());
        const NodeBlock* block = cache.getNodeBlockByID(nextBestUnvisitedNode.nodeID);
        visitNode(block, nextBestUnvisitedNode.path, nextBestUnvisitedNode.nodeID, nextBestUnvisitedNode.level,
                closedNodeQueue, debug_closedNodeQueue, currentSearchResults, queryImageBitCountMipmapStack, queryImage);

        // Re-sort search results
        std::sort(currentSearchResults.begin(), currentSearchResults.end());

        // Chop off irrelevant search results
        if(currentSearchResults.size() > resultCount) {
            currentSearchResults.erase(currentSearchResults.begin() + resultCount, currentSearchResults.end());
        }

        std::sort(debug_closedNodeQueue.begin(), debug_closedNodeQueue.end());

        /*std::cout << "Search results: ";
        for(int i = 0; i < currentSearchResults.size(); i++) {
            std::cout << currentSearchResults.at(i).distanceScore << ", ";
        }
        std::cout << std::endl;
        std::cout << "Closed nodes: ";
        for(int i = 0; i < debug_closedNodeQueue.size(); i++) {
            std::cout << debug_closedNodeQueue.at(i).minDistanceScore << "|" << debug_closedNodeQueue.at(i).nodeID << ", ";
        }
        std::cout << std::endl;*/
    }

    std::cout << "Query finished, " << computeMinDistanceThreshold(currentSearchResults) << " vs " << closedNodeQueue.top().minDistanceScore << std::endl;

    std::vector<SpinImage::index::QueryResult> queryResults;
    queryResults.reserve(resultCount);

    for(int i = 0; i < resultCount; i++) {
        queryResults.push_back({currentSearchResults.at(i).reference, currentSearchResults.at(i).image});
    }

    return queryResults;
}
