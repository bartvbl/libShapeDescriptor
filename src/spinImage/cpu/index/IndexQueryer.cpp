#include <queue>
#include "IndexQueryer.h"
#include <spinImage/cpu/index/types/IndexQueryItem.h>

std::stringstream IDBuilder;

std::vector<IndexEntry> queryIndex(Index index, unsigned int* queryImage, unsigned int resultCount) {
    std::vector<IndexQueryItem> queryQueue;
    queryQueue.reserve(resultCount);

    for(int i = 0; i < NODES_PER_BLOCK; i++) {
        if(index.rootNode.childNodeIsLeafNode[i] == true) {
            for(int entry = 0; entry < index.rootNode.leafNodeContentsLength[entry]; entry++) {
                queryQueue.emplace_back();
            }
        } else {
            IDBuilder.str(std::string());
            IDBuilder << std::hex << (i < 16 ? "0" : "") << i;
            queryQueue.emplace_back(IDBuilder.str());
        }
    }

    /*while(!scoreExceedsLimit()) {
        expandNextItem(queryQueue);
        updateScores(queryQueue);
        sortEntries(queryQueue);
        pruneQueue(queryQueue);
    }*/

    std::vector<IndexEntry> queryResults;
    queryResults.resize(resultCount);
    return queryResults;
}