#include <queue>
#include "IndexQueryer.h"
#include <spinImage/cpu/index/types/BitCountMipmapStack.h>
#include <algorithm>
#include <climits>
#include <cfloat>
#include <spinImage/cpu/index/types/IndexPath.h>
#include <set>

struct SearchResultEntry {
    SearchResultEntry(IndexEntry entry, const QuiccImage &imageEntry, std::string debug_nodeID, float minDistance)
        : reference(entry), debug_indexPath(debug_nodeID), distanceScore(minDistance) {}

    IndexEntry reference;
    std::string debug_indexPath;
    float distanceScore;

    bool operator< (const SearchResultEntry &right) const {
        return distanceScore < right.distanceScore;
    }
};

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

std::vector<SpinImage::index::QueryResult> SpinImage::index::query(Index &index, const QuiccImage &queryImage, unsigned int resultCount) {
    BitCountMipmapStack queryImageBitCountMipmapStack(queryImage);

    std::set<SearchResultEntry> currentSearchResults;

    // Root node path is not referenced, so can be left uninitialised
    IndexPath rootNodePath;

    std::cout << "Query finished" << std::endl;

    std::vector<SpinImage::index::QueryResult> queryResults;
    queryResults.reserve(resultCount);

    /*for(int i = 0; i < resultCount; i++) {
        queryResults.push_back({currentSearchResults.at(i).reference, currentSearchResults.at(i).image});
        std::cout << "Result " << i << ": "
               "file " << currentSearchResults.at(i).reference.fileIndex <<
               ", image " << currentSearchResults.at(i).reference.imageIndex <<
               ", score " << currentSearchResults.at(i).distanceScore <<
               ", path " << currentSearchResults.at(i).debug_indexPath << std::endl;
    }*/

    return queryResults;
}
