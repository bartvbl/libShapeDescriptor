#include <queue>
#include "IndexQueryer.h"
#include "NodeBlockCache.h"
#include <spinImage/cpu/index/types/IndexQueryItem.h>
#include <spinImage/cpu/index/types/BitCountMipmapStack.h>

struct UnvisitedNode {
    std::string indexNodeID;
    unsigned int minDistanceScore;
    unsigned int level;
};

struct SearchResultEntry {
    IndexEntry reference;
    MipMapLevel3 mipmapImage;
    unsigned int minDistanceScore;
    unsigned int level;
};

std::stringstream IDBuilder;

void visitNode(
        const NodeBlock* block,
        const unsigned int level,
        std::priority_queue<UnvisitedNode> &closedNodeQueue,
        std::vector<SearchResultEntry> &currentSearchResults,
        BitCountMipmapStack &queryImageMipmapStack) {
    expandNextItem(queryQueue);
    updateScores(queryQueue);
    sortEntries(queryQueue);
    pruneQueue(queryQueue);
}

std::vector<IndexEntry> queryIndex(Index index, unsigned int* queryImage, unsigned int resultCount) {
    BitCountMipmapStack queryImageMipmapStack(queryImage);

    NodeBlockCache cache(25000, index.indexDirectory, &index.rootNode);

    std::priority_queue<UnvisitedNode> closedNodeQueue;
    std::vector<SearchResultEntry> currentSearchResults;

    currentSearchResults.reserve(30000 + resultCount + NODES_PER_BLOCK * NODE_SPLIT_THRESHOLD);

    visitNode(&index.rootNode, 0, closedNodeQueue, currentSearchResults, queryImageMipmapStack);

    // Iteratively add additional nodes until there's no chance any additional node can improve the best distance score
    while(currentSearchResults.at(currentSearchResults.size() - 1).minDistanceScore > closedNodeQueue.top().minDistanceScore) {
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