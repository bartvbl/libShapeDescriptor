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

std::string childIndexToHexString(unsigned int childIndex) {
    const char hexDigits[17] = "0123456789ABCDEF";
    assert(childIndex < 256);
    return (childIndex < 16 ? "0" : "") + hexDigits[childIndex / 16] + hexDigits[childIndex % 16];
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
        const std::array<unsigned char, 8 + (16 * 16) + (32 * 32)> &haystack,
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
        } else if(level < 16 + 8) {
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
                needle.level3[(level - 8 - (2 * 16)) * 8 + 7],
        }

        // Compute and set partial min distance score
        unsigned int minLevelByteDistance = computeLevelByteMinDistance(bitDistances, levelByte);
        // Scores can only increase, and since we look at parts of columns at a time,
        partialScores[columnIndex] = std::max<unsigned short>(minLevelByteDistance, partialScores[columnIndex]);

        if(level < 8) {
            columnIndex++;
        } else if(level < (2 * 16) + 8) {
            if(level % 4 == 3) {
                columnIndex++;
            }
        } else {
            if(level % 16 == 15) {
                columnIndex++;
            }
        }
        nextPartialScoreIndex = nextPartialScoreIndex % 32;
    }

    // Finally, compute sum of partial sums
    unsigned int minDistanceSum = 0;
    for(unsigned short partialScore : partialScores) {
        minDistanceSum += partialScore;
    }
    return minDistanceSum;
}

void visitNode(
        const NodeBlock* block,
        const unsigned int level,
        std::priority_queue<UnvisitedNode> &closedNodeQueue,
        std::vector<SearchResultEntry> &currentSearchResults,
        const BitCountMipmapStack &queryImageMipmapStack,
        const QuiccImage &queryImage) {
    // Step 1: Divide child nodes over both queues
    // Simultaneously, compute distance scores
    const unsigned int childLevel = level + 1;
    const unsigned int searchResultScoreThreshold = currentSearchResults.at(currentSearchResults.size() - 1).distanceScore;
    for(int child = 0; child < NODES_PER_BLOCK; child++) {
        if(block->childNodeIsLeafNode[child] == true) {
            int nextNodeID = block->leafNodeContentsStartIndices[child];
            while(nextNodeID != -1) {
                NodeBlockEntry entry = block->leafNodeContents.at(nextNodeID);
                unsigned int distanceScore = computeDistance(queryImage, entry.image);

                if(distanceScore <= searchResultScoreThreshold) {
                    currentSearchResults.emplace_back(entry.indexEntry, distanceScore);
                }
                nextNodeID = entry.nextEntryIndex;
            }
        } else {
            unsigned int minDistanceScore = computeMinDistance(queryImageMipmapStack, )
        }
    }

    //updateScores(queryQueue);
    //sortEntries(queryQueue);
    //pruneQueue(queryQueue);
}

std::vector<IndexEntry> queryIndex(Index index, const QuiccImage &queryImage, unsigned int resultCount) {
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