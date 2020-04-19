#include <queue>
#include "IndexQueryer.h"
#include "Pattern.h"
#include <spinImage/cpu/index/types/BitCountMipmapStack.h>
#include <algorithm>
#include <set>
#include <unordered_map>
#include <spinImage/cpu/index/types/IndexEntry.h>
#include <fstream>
#include <fast-lzma2.h>

struct QueryPattern {
    unsigned int row;
    unsigned int col;
    unsigned int length;
};

struct HaystackPattern {
    unsigned int row;
    unsigned int col;
    unsigned int length;
    unsigned int overlapScore;
};



struct ImageReference {
    unsigned short matchingPixelCount = 0;
    unsigned short totalImagePixelCount = 0;
    IndexEntry entry;
};

struct SearchResultImageReference {
    float distanceScore = 0;
    IndexEntry entry;
};

// We're interpreting raw bytes as this struct.
// Need to disable padding bytes for that to work
#pragma pack(push, 1)
struct ListHeaderEntry {
    unsigned short pixelCount;
    unsigned int imageReferenceCount;
};
#pragma pack(pop)

struct IndexEntryListBuffer {
    std::array<unsigned int, spinImageWidthPixels * spinImageWidthPixels> startIndices;
    std::array<IndexEntry, 32> entryBuffer;
    unsigned int entryBufferLength = 0;
    unsigned int entryBufferPointer = 0;
    HaystackPattern pattern;

    void initialise(HaystackPattern haystackPattern, const std::experimental::filesystem::path &indexRootDirectory) {
        std::experimental::filesystem::path patternListFile = indexRootDirectory / "sorted_lists" / std::to_string(pattern.col) / std::to_string(pattern.row) / ("list_" + std::to_string(pattern.length) + ".dat");
        std::fstream listStream(patternListFile, std::ios::in | std::ios::binary);

        pattern = haystackPattern;

        char headerID[6] = {0, 0, 0, 0, 0, 0};
        listStream.read(headerID, 5);
        assert(std::string(headerID) == "PXLST");

        size_t indexEntryCount = 0xCDCDCDCDCDCDCDCD;
        unsigned short uniquePixelCountOccurrences = 0xCDCD;
        unsigned short compressedHeaderSize = 0xCDCD;
        size_t compressedBufferSize = 0xCDCDCDCDCDCDCDCD;

        listStream.read(reinterpret_cast<char *>(&indexEntryCount), sizeof(size_t));
        listStream.read(reinterpret_cast<char *>(&uniquePixelCountOccurrences), sizeof(unsigned short));
        listStream.read(reinterpret_cast<char *>(&compressedHeaderSize), sizeof(unsigned short));
        listStream.read(reinterpret_cast<char *>(&compressedBufferSize), sizeof(size_t));

        ListHeaderEntry* decompressedHeader = new ListHeaderEntry[uniquePixelCountOccurrences];
        char* compressedHeader = new char[compressedHeaderSize];
        listStream.read(compressedHeader, compressedHeaderSize);

        FL2_decompress(
                decompressedHeader, uniquePixelCountOccurrences * sizeof(ListHeaderEntry),
                compressedHeader, compressedHeaderSize);

        delete[] compressedHeader;



        delete[] decompressedHeader;

        listStream.close();
    }
};

std::pair<float, float> computePixelWeights(unsigned int queryImageSetBitCount) {
    const unsigned int bitsPerImage = spinImageWidthPixels * spinImageWidthPixels;
    unsigned int queryImageUnsetBitCount = bitsPerImage - queryImageSetBitCount;

    // If any count is 0, bump it up to 1
    queryImageSetBitCount = std::max<unsigned int>(queryImageSetBitCount, 1);
    queryImageUnsetBitCount = std::max<unsigned int>(queryImageUnsetBitCount, 1);

    // The fewer bits exist of a specific pixel type, the greater the penalty for not containing it
    float missedSetBitPenalty = float(bitsPerImage) / float(queryImageSetBitCount);
    float missedUnsetBitPenalty = float(bitsPerImage) / float(queryImageUnsetBitCount);

    return {missedSetBitPenalty, missedUnsetBitPenalty};
}

void findQueryPatterns(const QuiccImage &queryImage, std::array<std::vector<QueryPattern>, spinImageWidthPixels> &queryPatterns, unsigned int* totalPixelCount) {
    *totalPixelCount = 0;
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
                queryPatterns.at(col).push_back({patternStartRow, col, patternLength});
                *totalPixelCount += patternLength;
                patternLength = 0;
                patternStartRow = 0;
            }

            previousPixelWasSet = pixel == 1;
        }

        if (previousPixelWasSet) {
            queryPatterns.at(col).push_back({patternStartRow, col, patternLength});
            *totalPixelCount += patternLength;
        }
    }
}

void findHaystackPatterns(std::array<std::vector<QueryPattern>, spinImageWidthPixels> &queryPatterns, std::vector<HaystackPattern> &haystackPatterns) {
    for (unsigned int col = 0; col < spinImageWidthPixels; col++) {
        for (unsigned int row = 0; row < spinImageWidthPixels; row++) {
            unsigned int maxPatternLength = spinImageWidthPixels - row;
            for(unsigned int patternLength = 0; patternLength < maxPatternLength; patternLength++) {
                unsigned int overlapScore = 0;
                unsigned int totalQueryLength = 0;

                for(unsigned int queryPatternIndex = 0; queryPatternIndex < queryPatterns.at(col).size(); queryPatternIndex++) {
                    QueryPattern queryPattern = queryPatterns.at(col).at(queryPatternIndex);
                    unsigned int queryStartIndex = queryPattern.row;
                    unsigned int queryEndIndex = queryPattern.row + queryPattern.length - 1;
                    unsigned int haystackStartIndex = row;
                    unsigned int haystackEndIndex = row + patternLength - 1;
                    totalQueryLength += queryPattern.length;

                    // No overlap
                    if(queryEndIndex < haystackStartIndex || queryStartIndex > haystackEndIndex) {
                        continue;
                    }

                    unsigned int overlapStart = std::max(queryStartIndex, haystackStartIndex);
                    unsigned int overlapEnd = std::min(queryEndIndex, haystackEndIndex);

                    overlapScore += overlapEnd - overlapStart + 1;
                }

                if(overlapScore != 0
                /* IN CASE OF EMERGENCY UNCOMMENT THIS && overlapScore == totalQueryLength*/) {
                    haystackPatterns.push_back({row, col, patternLength, overlapScore});
                }
            }
        }
    }
}

std::vector<SpinImage::index::QueryResult> SpinImage::index::query(Index &index, const QuiccImage &queryImage, unsigned int resultCount) {
    std::array<std::vector<QueryPattern>, spinImageWidthPixels> queryPatterns;
    std::vector<HaystackPattern> haystackPatterns;

    unsigned int queryPixelCount = 0;
    findQueryPatterns(queryImage, queryPatterns, &queryPixelCount);
    findHaystackPatterns(queryPatterns, haystackPatterns);
    std::cout << haystackPatterns.size() << " haystack patterns" << std::endl;

    std::vector<IndexEntryListBuffer> patternBuffers;
    patternBuffers.resize(haystackPatterns.size());

    std::pair<float, float> combinedPixelWeights = computePixelWeights(queryPixelCount);
    float missingSetPixelPenalty = combinedPixelWeights.first;
    float missingUnsetPixelPenalty = combinedPixelWeights.second;
    std::cout << "Pixel weights: " << missingSetPixelPenalty << "/missing set, " << missingUnsetPixelPenalty << "/missing unset" << std::endl;

    std::set<SearchResultImageReference> results;

#pragma omp parallel
    {
        #pragma omp for
        for(unsigned int haystackPatternIndex = 0; haystackPatternIndex < haystackPatterns.size(); haystackPatternIndex++) {
            patternBuffers.at(haystackPatternIndex).initialise(haystackPatterns.at(haystackPatternIndex), index.indexDirectory);
        }

        for(int totalImagePixelCount = 1; totalImagePixelCount < spinImageWidthPixels * spinImageWidthPixels; totalImagePixelCount++) {



        }
    }

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

