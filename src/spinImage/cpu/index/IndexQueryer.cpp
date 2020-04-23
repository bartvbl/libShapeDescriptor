#include <queue>
#include "IndexQueryer.h"
#include "Pattern.h"
#include <spinImage/cpu/index/types/BitCountMipmapStack.h>
#include <algorithm>
#include <set>
#include <unordered_map>
#include <spinImage/cpu/index/types/IndexEntry.h>
#include <spinImage/cpu/types/BoolArray.h>
#include <fstream>
#include <chrono>
#include <thread>
#include <fast-lzma2.h>
#include <spinImage/cpu/index/types/ListHeaderEntry.h>

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

struct ReferenceMetadata {
    unsigned short matchingPixelCount = 0;
    unsigned short totalImagePixelCount = 0;
};

struct ImageReference {
    union multipurpose {
        ReferenceMetadata asMetadata;
        float asDistanceScore = 0;

        multipurpose() {}
    } multipurpose;
    IndexEntry entry;
};

bool sortByImageIndexComparator(const ImageReference& lhs, const ImageReference& rhs) {
    if(lhs.multipurpose.asMetadata.totalImagePixelCount != rhs.multipurpose.asMetadata.totalImagePixelCount) {
        return lhs.multipurpose.asMetadata.totalImagePixelCount < rhs.multipurpose.asMetadata.totalImagePixelCount;
    }

    return lhs.entry < rhs.entry;
}

bool sortByDistanceScoreComparator (const ImageReference& lhs, const ImageReference& rhs) {
    return lhs.multipurpose.asDistanceScore < rhs.multipurpose.asDistanceScore;
}

struct SearchResultImageReference {
    float distanceScore = 0;
    IndexEntry entry;
};

struct IndexEntryListBuffer {
    std::vector<ImageReference> sortedIndexEntryList;
    HaystackPattern pattern;
    size_t copyBufferBaseIndex = 0;
    size_t mergeBufferSize = 0;

    void initialise(HaystackPattern haystackPattern, const std::experimental::filesystem::path &indexRootDirectory) {
        pattern = haystackPattern;

        std::experimental::filesystem::path patternListFile = indexRootDirectory / "sorted_lists" / std::to_string(pattern.col) / std::to_string(pattern.row) / ("list_" + std::to_string(pattern.length) + ".dat");
        std::fstream listStream(patternListFile, std::ios::in | std::ios::binary);

        char headerID[6] = {0, 0, 0, 0, 0, 0};
        listStream.read(headerID, 5);
        assert(std::string(headerID) == "PXLST");

        size_t indexEntryCount = 0xCDCDCDCDCDCDCDCD;
        unsigned short headerEntryCount = 0xCDCD;
        unsigned short compressedHeaderSize = 0xCDCD;
        size_t compressedBufferSize = 0xCDCDCDCDCDCDCDCD;

        listStream.read(reinterpret_cast<char *>(&indexEntryCount), sizeof(size_t));
        listStream.read(reinterpret_cast<char *>(&headerEntryCount), sizeof(unsigned short));
        listStream.read(reinterpret_cast<char *>(&compressedHeaderSize), sizeof(unsigned short));
        listStream.read(reinterpret_cast<char *>(&compressedBufferSize), sizeof(size_t));

        //std::cout << "Header: " + std::to_string(indexEntryCount) + ", " + std::to_string(headerEntryCount) + ", " + std::to_string(compressedHeaderSize) + ", " + std::to_string(compressedBufferSize) + " -> " + patternListFile.string() + "\n";

        if(headerEntryCount == 0) {
            // Nothing to be done if file is empty
            listStream.close();
            return;
        }

        // Read header

        ListHeaderEntry* decompressedHeader = new ListHeaderEntry[headerEntryCount];
        char* compressedBuffer = new char[std::max<size_t>(compressedHeaderSize, compressedBufferSize)];
        listStream.read(compressedBuffer, compressedHeaderSize);

        FL2_decompress(
                decompressedHeader, headerEntryCount * sizeof(ListHeaderEntry),
                compressedBuffer, compressedHeaderSize);

        // Read file contents

        sortedIndexEntryList.resize(indexEntryCount);

        std::vector<IndexEntry> indexEntryList;
        indexEntryList.resize(indexEntryCount);

        listStream.read(compressedBuffer, compressedBufferSize);
        listStream.close();

        FL2_decompress(
                indexEntryList.data(), indexEntryCount * sizeof(IndexEntry),
                compressedBuffer, compressedBufferSize);

        delete[] compressedBuffer;

        // Expand file contents

        unsigned short currentPixelCount = decompressedHeader[0].pixelCount;
        unsigned short currentHeaderEntryIndex = 0;
        unsigned int nextPixelCountIncrementIndex = decompressedHeader[0].imageReferenceCount;

        for(unsigned int i = 0; i < indexEntryCount; i++) {
            if(i >= nextPixelCountIncrementIndex) {
                currentHeaderEntryIndex++;
                assert(currentHeaderEntryIndex < headerEntryCount);
                currentPixelCount = decompressedHeader[currentHeaderEntryIndex].pixelCount;
                nextPixelCountIncrementIndex += decompressedHeader[currentHeaderEntryIndex].imageReferenceCount;
            }
            IndexEntry entry = indexEntryList.at(i);
            sortedIndexEntryList.at(i).multipurpose.asMetadata = {(unsigned short) pattern.overlapScore, currentPixelCount};
            sortedIndexEntryList.at(i).entry = entry;
        }

        delete[] decompressedHeader;
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

void findHaystackPatterns(std::array<std::vector<QueryPattern>, spinImageWidthPixels> &queryPatterns, std::vector<HaystackPattern>* haystackPatterns) {
    for (unsigned int col = 0; col < spinImageWidthPixels; col++) {
        for (unsigned int row = 0; row < spinImageWidthPixels; row++) {
            unsigned int maxPatternLength = spinImageWidthPixels - row;
            for(unsigned int patternLength = 1; patternLength < maxPatternLength; patternLength++) {
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
                    haystackPatterns->push_back({row, col, patternLength, overlapScore});
                }
            }
        }
    }
}

std::vector<SpinImage::index::QueryResult> SpinImage::index::query(Index &index, const QuiccImage &queryImage, unsigned int resultCount) {
    std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();

    std::array<std::vector<QueryPattern>, spinImageWidthPixels> queryPatterns;
    std::vector<HaystackPattern> haystackPatterns;

    unsigned int queryPixelCount = 0;
    findQueryPatterns(queryImage, queryPatterns, &queryPixelCount);
    findHaystackPatterns(queryPatterns, &haystackPatterns);
    std::cout << haystackPatterns.size() << " haystack patterns" << std::endl;

    std::pair<float, float> combinedPixelWeights = computePixelWeights(queryPixelCount);
    float missingSetPixelPenalty = combinedPixelWeights.first;
    float missingUnsetPixelPenalty = combinedPixelWeights.second;
    std::cout << "Pixel weights: " << missingSetPixelPenalty << "/missing set, " << missingUnsetPixelPenalty << "/missing unset" << std::endl;

    // ----------------------------------------------------------

    std::cout << "Reading list files.. " << std::flush;
    std::chrono::steady_clock::time_point listReadingStartTime = std::chrono::steady_clock::now();
        std::vector<IndexEntryListBuffer> patternBuffers;
        patternBuffers.resize(haystackPatterns.size());
        #pragma omp parallel for schedule(dynamic)
        for(unsigned int haystackPatternIndex = 0; haystackPatternIndex < haystackPatterns.size(); haystackPatternIndex++) {
            patternBuffers.at(haystackPatternIndex).initialise(haystackPatterns.at(haystackPatternIndex), index.indexDirectory);
        }
    std::chrono::steady_clock::time_point listReadingEndTime = std::chrono::steady_clock::now();
    auto listReadingDuration = std::chrono::duration_cast<std::chrono::milliseconds>(listReadingEndTime - listReadingStartTime);
    std::cout << "duration: " << float(listReadingDuration.count()) / 1000.0f << " seconds" << std::endl;

    // ----------------------------------------------------------

    std::cout << "Counting lists.." << std::endl;
    size_t totalIndexEntryCount = 0;
    for(unsigned int haystackPatternIndex = 0; haystackPatternIndex < haystackPatterns.size(); haystackPatternIndex++) {
        size_t elementsInPatternList = patternBuffers.at(haystackPatternIndex).sortedIndexEntryList.size();
        patternBuffers.at(haystackPatternIndex).copyBufferBaseIndex = totalIndexEntryCount;
        patternBuffers.at(haystackPatternIndex).mergeBufferSize = elementsInPatternList;
        //std::cout << std::to_string(totalIndexEntryCount) + ", " + std::to_string(elementsInPatternList) + "\n";
        totalIndexEntryCount += elementsInPatternList;
    }

    // ----------------------------------------------------------

    std::cout << "Copying buffers.. " << std::flush;
    std::chrono::steady_clock::time_point bufferCopyStartTime = std::chrono::steady_clock::now();
        std::vector<ImageReference> indexEntryList;
        indexEntryList.resize(totalIndexEntryCount);
        #pragma omp parallel for schedule(dynamic)
        for(unsigned int haystackPatternIndex = 0; haystackPatternIndex < haystackPatterns.size(); haystackPatternIndex++) {
            std::copy(patternBuffers.at(haystackPatternIndex).sortedIndexEntryList.begin(),
                    patternBuffers.at(haystackPatternIndex).sortedIndexEntryList.end(),
                    indexEntryList.begin() + patternBuffers.at(haystackPatternIndex).copyBufferBaseIndex);
        }
    std::chrono::steady_clock::time_point bufferCopyEndTime = std::chrono::steady_clock::now();
    auto bufferCopyDuration = std::chrono::duration_cast<std::chrono::milliseconds>(bufferCopyEndTime - bufferCopyStartTime);
    std::cout << "duration: " << float(bufferCopyDuration.count()) / 1000.0f << " seconds" << std::endl;

    // ----------------------------------------------------------

    // Sort by image reference
    std::cout << "First sort.. " << std::flush;
    std::chrono::steady_clock::time_point firstSortStartTime = std::chrono::steady_clock::now();
        for(unsigned int mergeDistance = 2; mergeDistance < 2 * patternBuffers.size(); mergeDistance *= 2) {
            #pragma omp parallel for schedule(dynamic)
            for(unsigned int haystackPatternIndex = 0; haystackPatternIndex < patternBuffers.size(); haystackPatternIndex += mergeDistance) {
                unsigned int mergeListIndex = haystackPatternIndex + mergeDistance / 2;
                if(mergeListIndex >= patternBuffers.size()) {
                    continue;
                }

                size_t begin = patternBuffers.at(haystackPatternIndex).copyBufferBaseIndex;
                size_t middle = begin + patternBuffers.at(haystackPatternIndex).mergeBufferSize;
                size_t end = middle + patternBuffers.at(mergeListIndex).mergeBufferSize;

                std::inplace_merge(
                    indexEntryList.begin() + begin,
                    indexEntryList.begin() + middle,
                    indexEntryList.begin() + end,
                    sortByImageIndexComparator);

                patternBuffers.at(haystackPatternIndex).mergeBufferSize += patternBuffers.at(mergeListIndex).mergeBufferSize;
            }
        }
    std::chrono::steady_clock::time_point firstSortEndTime = std::chrono::steady_clock::now();
    auto firstSortDuration = std::chrono::duration_cast<std::chrono::milliseconds>(firstSortEndTime - firstSortStartTime);
    std::cout << "duration: " << float(firstSortDuration.count()) / 1000.0f << " seconds" << std::endl;

    // ----------------------------------------------------------

    std::cout << "Merging references.." << std::endl;
    size_t targetMergeIndex = 0;
    for(size_t sourceMergeIndex = 0; sourceMergeIndex < totalIndexEntryCount;) {
        ImageReference mergedReference = indexEntryList.at(sourceMergeIndex);
        sourceMergeIndex++;
        // Merge next references until the end of the list or until the current reference index changes
        while(sourceMergeIndex < totalIndexEntryCount &&
            indexEntryList.at(sourceMergeIndex).entry == mergedReference.entry) {
            ImageReference nextReference = indexEntryList.at(sourceMergeIndex);
            mergedReference.multipurpose.asMetadata.matchingPixelCount += nextReference.multipurpose.asMetadata.matchingPixelCount;
            sourceMergeIndex++;
        }

        unsigned int missingSetPixelCount = queryPixelCount - mergedReference.multipurpose.asMetadata.matchingPixelCount;
        unsigned int missingUnsetPixelCount = mergedReference.multipurpose.asMetadata.totalImagePixelCount - mergedReference.multipurpose.asMetadata.matchingPixelCount;

        float distanceScore =
                float(missingSetPixelCount) * missingSetPixelPenalty +
                float(missingUnsetPixelCount) * missingUnsetPixelPenalty;

        mergedReference.multipurpose.asDistanceScore = distanceScore;

        indexEntryList.at(targetMergeIndex) = mergedReference;
        targetMergeIndex++;
    }

    // Throw away remainder
    indexEntryList.resize(targetMergeIndex);

    // Sort by distance score
    std::cout << "Second sort.." << std::endl;
    std::sort(indexEntryList.begin(), indexEntryList.end(), sortByDistanceScoreComparator);

    std::chrono::steady_clock::time_point endTime = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << std::endl << "Query complete. " << std::endl;
    std::cout << "Total execution time: " << float(duration.count()) / 1000.0f << " seconds" << std::endl;

    std::vector<SpinImage::index::QueryResult> queryResults;
    queryResults.reserve(resultCount);

    for(int i = 0; i < resultCount; i++) {
        queryResults.push_back({indexEntryList.at(i).entry, indexEntryList.at(i).multipurpose.asDistanceScore});
        std::cout << "Result " << i
        << ": score " << indexEntryList.at(i).multipurpose.asDistanceScore
        << ", file " << indexEntryList.at(i).entry.fileIndex
        << ", image " << indexEntryList.at(i).entry.imageIndex << std::endl;
    }

    return queryResults;
}

