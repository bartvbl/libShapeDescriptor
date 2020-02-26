#include <queue>
#include "IndexQueryer.h"
#include "NodeBlockCache.h"
#include <spinImage/cpu/index/types/BitCountMipmapStack.h>

struct UnvisitedNode {
    std::string indexNodeID;
    unsigned int minDistanceScore;
    unsigned int level;
};

struct SearchResultEntry {
    SearchResultEntry(IndexEntry entry, unsigned int minDistance)
        : reference(entry), distanceScore(minDistance) {}

    const IndexEntry reference;
    const unsigned int distanceScore;
};

std::stringstream IDBuilder;

/*std::string childIndexToHexString(int childIndex) {
    return (childIndex < 16 ? "0" : "") + "";
}*/

void visitNode(
        const NodeBlock* block,
        const unsigned int level,
        std::priority_queue<UnvisitedNode> &closedNodeQueue,
        std::vector<SearchResultEntry> &currentSearchResults,
        BitCountMipmapStack &queryImageMipmapStack) {
    // Step 1: Divide child nodes over both queues
    // Simultaneously, compute distance scores
    const unsigned int childLevel = level + 1;
    const unsigned int searchResultScoreThreshold = currentSearchResults.at(currentSearchResults.size() - 1).distanceScore;
    for(int child = 0; child < NODES_PER_BLOCK; child++) {
        if(block->childNodeIsLeafNode[child] == true) {
            int nextNodeID = block->leafNodeContentsStartIndices[child];
            while(nextNodeID != -1) {
                NodeBlockEntry entry = block->leafNodeContents.at(nextNodeID);
                unsigned int distanceScore = computeMinimumDistance(entry.mipmapImage, queryImageMipmapStack, childLevel);

                if(distanceScore <= searchResultScoreThreshold) {
                    currentSearchResults.emplace_back(entry.indexEntry, distanceScore);
                }
                nextNodeID = entry.nextEntryIndex;
            }
        } else {

        }
    }

    //updateScores(queryQueue);
    //sortEntries(queryQueue);
    //pruneQueue(queryQueue);
}

std::vector<IndexEntry> queryIndex(Index index, unsigned int* queryImage, unsigned int resultCount) {
    BitCountMipmapStack queryImageMipmapStack(queryImage);

    NodeBlockCache cache(25000, index.indexDirectory, &index.rootNode);

    std::priority_queue<UnvisitedNode> closedNodeQueue;
    std::vector<SearchResultEntry> currentSearchResults;

    currentSearchResults.reserve(30000 + resultCount + NODES_PER_BLOCK * NODE_SPLIT_THRESHOLD);

    visitNode(&index.rootNode, 0, closedNodeQueue, currentSearchResults, queryImageMipmapStack);

    // Iteratively add additional nodes until there's no chance any additional node can improve the best distance score
    while(currentSearchResults.at(currentSearchResults.size() - 1).distanceScore > closedNodeQueue.top().minDistanceScore) {
        UnvisitedNode nextBestUnvisitedNode = closedNodeQueue.top();
        closedNodeQueue.pop();
        const NodeBlock* block = cache.fetch(nextBestUnvisitedNode.indexNodeID);
        visitNode(block, nextBestUnvisitedNode.level, closedNodeQueue, currentSearchResults, queryImageMipmapStack);
    }

    std::vector<IndexEntry> queryResults;
    queryResults.reserve(resultCount);

    for(int i = 0; i < resultCount; i++) {
        queryResults.push_back(currentSearchResults.at(i).reference);
    }

    return queryResults;
}