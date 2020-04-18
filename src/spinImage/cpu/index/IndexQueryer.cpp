#include <queue>
#include "IndexQueryer.h"
#include "Pattern.h"
#include <spinImage/cpu/index/types/BitCountMipmapStack.h>
#include <algorithm>
#include <set>
#include <unordered_map>
#include <spinImage/cpu/index/types/IndexEntry.h>

struct Pattern {
    unsigned int row;
    unsigned int col;
    unsigned int length;
};

struct ImageReference {
    unsigned short matchingPixelCount;
    IndexEntry entry;
};

struct IndexEntryListBuffer {
    //const float scorePenalty;
    std::array<IndexEntry, 32> entryBuffer;
    unsigned int entryBufferLength = 0;
    unsigned int entryBufferPointer = 0;

    /*IndexEntryListBuffer(Pattern pattern, std::experimental::filesystem::path indexRootDirectory) {

    }*/

    IndexEntry top() {

    }

    void advance() {

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

std::vector<SpinImage::index::QueryResult> SpinImage::index::query(Index &index, const QuiccImage &queryImage, unsigned int resultCount) {
    std::vector<Pattern> queryPatterns;

    for (unsigned int col = 0; col < spinImageWidthPixels; col++) {
        unsigned int patternLength = 0;
        unsigned int patternStartRow = 0;
        bool previousPixelWasSet = false;

        for (unsigned int row = 0; row < spinImageWidthPixels; row++) {
            int pixel = SpinImage::index::pattern::pixelAt(queryImage, row, col);
            if (pixel == 1) {
                if (previousPixelWasSet) {
                    // Pattern turned out to be one pixel longer
                    patternLength++;
                } else {
                    // We found a new pattern
                    patternStartRow = row;
                    patternLength = 1;
                }
            } else if (previousPixelWasSet) {
                // Previous pixel was set, but this one is not
                // This is thus a pattern that ended here.
                queryPatterns.push_back({patternStartRow, col, patternLength});
                patternLength = 0;
                patternStartRow = 0;
            }

            previousPixelWasSet = pixel == 1;
        }

        if (previousPixelWasSet) {
            queryPatterns.push_back({patternStartRow, col, patternLength});
        }
    }

    for(const Pattern &pattern : queryPatterns) {
        std::cout << "Pattern: " << pattern.col << ", " << pattern.row << ", length " << pattern.length << std::endl;
    }

    //std::vector<

#pragma omp parallel
    {
        //std::vector<
        for(int totalImagePixelCount = 1; totalImagePixelCount < spinImageWidthPixels * spinImageWidthPixels; totalImagePixelCount++) {
            //unsigned int

            // Step 1: Read chunks
            /*#pragma omp for
            for(IndexEntryListBuffer &patternBuffer : patternBuffers) {

            }*/

            // Step 2: Compute total number of entries

            // Step 3: Copy entries into single list

            // Step 4: Sort list

            // Step 5: Merge images together into one entry


        }
    };

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
