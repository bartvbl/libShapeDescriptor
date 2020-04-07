#include <queue>
#include "IndexQueryer.h"
#include <spinImage/cpu/index/types/BitCountMipmapStack.h>
#include <algorithm>
#include <climits>
#include <cfloat>
#include <spinImage/cpu/index/types/IndexPath.h>

struct UnvisitedNode {
    UnvisitedNode(IndexPath indexPath, std::string unvisitedNodeID, float minDistance, unsigned int nodeLevel)
    : path(indexPath), nodeID(unvisitedNodeID), minDistanceScore(minDistance), level(nodeLevel) {}

    IndexPath path;
    std::string nodeID;
    float minDistanceScore;
    unsigned int level;

    // We want the open node priority queue to sort items by lowest score
    // Since the priority queue by default optimises for finding the highest sorted element,
    // we need to invert the sort order.
    bool operator< (const UnvisitedNode &right) const {
        if(minDistanceScore != right.minDistanceScore) {
            return minDistanceScore > right.minDistanceScore;
        }
        return level < right.level;
    }
};

struct SearchResultEntry {
    SearchResultEntry(IndexEntry entry, const QuiccImage &imageEntry, std::string debug_nodeID, float minDistance)
        : reference(entry), image(imageEntry), debug_indexPath(debug_nodeID), distanceScore(minDistance) {}

    IndexEntry reference;
    QuiccImage image;
    std::string debug_indexPath;
    float distanceScore;

    bool operator< (const SearchResultEntry &right) const {
        return distanceScore < right.distanceScore;
    }
};

std::stringstream IDBuilder;
unsigned int debug_visitedNodeCount = 0;

std::string appendPath(const std::string &parentNodeID, unsigned long childIndex) {
    std::string byteString = parentNodeID;
    const std::string characterMap = "0123456789abcdef";
    byteString += characterMap.at((childIndex >> 12U) & 0x0FU);
    byteString += characterMap.at((childIndex >> 8U) & 0x0FU);
    byteString += characterMap.at((childIndex >> 4U) & 0x0FU);
    byteString += characterMap.at((childIndex & 0x0FU));
    byteString += "/";
    return byteString;

}

float computeWeightedHammingDistance(const QuiccImage &needle, const BitCountMipmapStack &needleMipmapStack, const QuiccImage &haystack) {
    const unsigned int bitsPerImage = spinImageWidthPixels * spinImageWidthPixels;
    unsigned int queryImageSetBitCount = needleMipmapStack.level1[0] + needleMipmapStack.level1[1]
            + needleMipmapStack.level1[2] + needleMipmapStack.level1[3];
    unsigned int queryImageUnsetBitCount = bitsPerImage - queryImageSetBitCount;

    // If any count is 0, bump it up to 1
    queryImageSetBitCount = std::max<unsigned int>(queryImageSetBitCount, 1);
    queryImageUnsetBitCount = std::max<unsigned int>(queryImageUnsetBitCount, 1);

    // The fewer bits exist of a specific pixel type, the greater the penalty for not containing it
    float missedSetBitPenalty = float(bitsPerImage) / float(queryImageSetBitCount);
    float missedUnsetBitPenalty = float(bitsPerImage) / float(queryImageUnsetBitCount);

    // Wherever pixels don't match, we apply a penalty for each of them
    float score = 0;
    for(int i = 0; i < needle.size(); i++) {
        unsigned int wrongSetBitCount = std::bitset<32>((needle[i] ^ haystack[i]) & needle[i]).count();
        unsigned int wrongUnsetBitCount = std::bitset<32>((~needle[i] ^ ~haystack[i]) & ~needle[i]).count();
        score += float(wrongSetBitCount) * missedSetBitPenalty + float(wrongUnsetBitCount) * missedUnsetBitPenalty;
    }

    return score;
}

const float computeMinDistanceThreshold(std::vector<SearchResultEntry> &currentSearchResults) {
    return currentSearchResults.empty() ?
               std::numeric_limits<float>::max()
               : currentSearchResults.at(currentSearchResults.size() - 1).distanceScore;
}

void visitNode(
        const NodeBlock* block,
        IndexPath path,
        const std::string &nodeID,
        const unsigned int level,
        std::priority_queue<UnvisitedNode> &closedNodeQueue,
        std::vector<SearchResultEntry> &currentSearchResults,
        const BitCountMipmapStack &queryImageMipmapStack,
        const QuiccImage &queryImage) {
    // Divide child nodes over both queues
    const unsigned int childLevel = level + 1;
    // If we have not yet acquired any search results, disable the threshold
    const float searchResultScoreThreshold =
            computeMinDistanceThreshold(currentSearchResults);


    std::cout << "Visiting node " << debug_visitedNodeCount << " -> " << currentSearchResults.size() << " search results, " << closedNodeQueue.size() << " queued nodes, " << searchResultScoreThreshold  << " vs " << closedNodeQueue.top().minDistanceScore << " - " << nodeID << std::endl;
    for(int child = 0; child < NODES_PER_BLOCK; child++) {
        if(block->childNodeIsLeafNode[child]) {
            //std::cout << "Child " << child << " is leaf node!" << std::endl;
            // If child is a leaf node, insert its images into the search result list
            for(const NodeBlockEntry& entry : block->leafNodeContents.at(child)) {
                float distanceScore = computeWeightedHammingDistance(queryImage, queryImageMipmapStack, entry.image);

                // Only consider the image if it is potentially better than what's there already
                if(distanceScore <= searchResultScoreThreshold) {
                    currentSearchResults.emplace_back(entry.indexEntry, entry.image, nodeID, distanceScore);
                }
            }
        } else {
            // If the child is an intermediate node, enqueue it in the closed node list
            IndexPath childPath = path.append(child);
            float minDistanceScore = childPath.computeMinDistanceTo(queryImageMipmapStack);

            if(minDistanceScore <= searchResultScoreThreshold) {
                //unsigned int hammingDistance = computeHammingDistance(queryImageMipmapStack, childPath, level);
                //std::cout << "Enqueued " << appendPath(nodeID, child) << " -> " << minDistanceScore << std::endl;
                //std::cout << "Enqueued " << appendPath(nodeID, child) << " -> " << minDistanceScore << std::endl;
                closedNodeQueue.emplace(
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

    std::priority_queue<UnvisitedNode> closedNodeQueue;
    std::vector<SearchResultEntry> currentSearchResults;

    currentSearchResults.reserve(30000 + resultCount + NODES_PER_BLOCK * NODE_SPLIT_THRESHOLD);

    // Root node path is not referenced, so can be left uninitialised
    IndexPath rootNodePath;
    closedNodeQueue.emplace(rootNodePath, "", 0, 0);
    debug_visitedNodeCount = 0;

    // Iteratively add additional nodes until there's no chance any additional node can improve the best distance score
    while(  !closedNodeQueue.empty() &&
            computeMinDistanceThreshold(currentSearchResults) >= closedNodeQueue.top().minDistanceScore) {
        UnvisitedNode nextBestUnvisitedNode = closedNodeQueue.top();
        closedNodeQueue.pop();
        //visitNode(block, nextBestUnvisitedNode.path, nextBestUnvisitedNode.nodeID, nextBestUnvisitedNode.level,
        //        closedNodeQueue, currentSearchResults, queryImageBitCountMipmapStack, queryImage);
        debug_visitedNodeCount++;

        // Re-sort search results
        std::sort(currentSearchResults.begin(), currentSearchResults.end());

        // Chop off irrelevant search results
        if(currentSearchResults.size() > resultCount) {
            currentSearchResults.erase(currentSearchResults.begin() + resultCount, currentSearchResults.end());
        }

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
        std::cout << "Result " << i << ": "
               "file " << currentSearchResults.at(i).reference.fileIndex <<
               ", image " << currentSearchResults.at(i).reference.imageIndex <<
               ", score " << currentSearchResults.at(i).distanceScore <<
               ", path " << currentSearchResults.at(i).debug_indexPath << std::endl;
    }

    return queryResults;
}
