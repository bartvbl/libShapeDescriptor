#include <spinImage/utilities/fileutils.h>
#include <iostream>
#include <spinImage/cpu/types/QUICCIImages.h>
#include <spinImage/utilities/readers/quicciReader.h>
#include <bitset>
#include <spinImage/cpu/types/QuiccImage.h>
#include <json.hpp>
#include <fstream>
#include "IndexBuilder.h"
#include "tsl/ordered_map.h"
#include "IndexIO.h"

#include <fast-lzma2.h>
#include <malloc.h>
#include <spinImage/cpu/index/types/IndexEntry.h>

template<class Key, class T, class Ignore, class Allocator,
        class Hash = std::hash<Key>, class KeyEqual = std::equal_to<Key>,
        class AllocatorPair = typename std::allocator_traits<Allocator>::template rebind_alloc<std::pair<Key, T>>,
        class ValueTypeContainer = std::vector<std::pair<Key, T>, AllocatorPair>>
using ordered_map = tsl::ordered_map<Key, T, Hash, KeyEqual, AllocatorPair, ValueTypeContainer>;

using json = nlohmann::basic_json<ordered_map>;

// Phase 1:
//      - Create all pattern files, dump references into them in arbitrary order
//          -> Should only be dependent on number of patterns being tested for.
//             Because we write stuff immediately to disk, we don't keep much in memory
// Phase 2:
//      - Within each pattern file, sort all references by remaining pixel count (low to high)
//          -> Probably requires loading in an entire file at once due to compression
//      - Modify the header to represent which child patterns exist
//      - Create shortcut links for patterns where no direct link exists


struct IndexedFileStatistics {
    // General
    std::string filePath;
    size_t imageCount;
    size_t fileIndex;
    size_t cachedNodeBlockCount;
    size_t cachedImageCount;
    double totalExecutionTimeMilliseconds;
    double totalLinearInsertionTimeMilliseconds;

    // Cache
    size_t cacheMisses;
    size_t cacheHits;
    size_t cacheEvictions;
    size_t cacheDirtyEvictions;
    size_t cacheInsertions;

    // Node Block Cache specific
    size_t imageInsertionCount;
    size_t nodeBlockSplitCount;
    size_t nodeBlockReadCount;
    size_t nodeBlockWriteCount;
    double totalReadTimeMilliseconds;
    double totalWriteTimeMilliseconds;
    double totalSplitTimeMilliseconds;

    // Various cache related statistics
    unsigned long totalAllocatedImageCapacity;
    unsigned long totalCachedImageCount;
    unsigned long cachedLeafNodeCount;
    unsigned long cachedIntermediateNodeCount;
    unsigned long maximumImagesPerNode;
    unsigned long maximumImagesPerNodeBlock;
};

struct IndexConstructionSettings {
    std::experimental::filesystem::path quicciImageDumpDirectory;
    std::experimental::filesystem::path indexDumpDirectory;
    size_t cacheNodeBlockCapacity;
    size_t cacheImageCapacity;
    size_t fileStartIndex;
    size_t fileEndIndex;
};

IndexedFileStatistics gatherFileStatistics(
        size_t fileIndex,
        double totalImageDurationMilliseconds,
        double totalExecutionTimeMilliseconds,
        size_t imageCount,
        const std::string &filePath) {

    IndexedFileStatistics stats;
    stats.filePath = filePath;
    stats.fileIndex = fileIndex;
    stats.imageCount = imageCount;
    stats.totalExecutionTimeMilliseconds = totalExecutionTimeMilliseconds;
    stats.totalLinearInsertionTimeMilliseconds = totalImageDurationMilliseconds;

    return stats;
}

void dumpStatisticsFile(
        const std::vector<IndexedFileStatistics> &fileStatistics,
        const IndexConstructionSettings &constructionSettings,
        const std::experimental::filesystem::path &path) {
    json outJson;

    outJson["version"] = "v3";
    outJson["nodesPerBlock"] = NODES_PER_BLOCK;
    outJson["nodeSplitThreshold"] = NODE_SPLIT_THRESHOLD;
    outJson["cacheNodeBlockCapacity"] = constructionSettings.cacheNodeBlockCapacity;
    outJson["cacheImageCapacity"] = constructionSettings.cacheImageCapacity;
    outJson["outputIndexDirectory"] = constructionSettings.indexDumpDirectory;
    outJson["inputImageDirectory"] = constructionSettings.quicciImageDumpDirectory;
    outJson["fileStartIndex"] = constructionSettings.fileStartIndex;
    outJson["fileEndIndex"] = constructionSettings.fileEndIndex;

    outJson["fileStats"] = {};
    for(const IndexedFileStatistics &fileStats : fileStatistics) {
        outJson["fileStats"][fileStats.filePath] = {};

        outJson["fileStats"][fileStats.filePath]["fileIndex"] = fileStats.fileIndex;
        outJson["fileStats"][fileStats.filePath]["imageCount"] = fileStats.imageCount;
        outJson["fileStats"][fileStats.filePath]["cachedNodeBlockCount"] = fileStats.cachedNodeBlockCount;
        outJson["fileStats"][fileStats.filePath]["cachedImageCount"] = fileStats.cachedImageCount;
        outJson["fileStats"][fileStats.filePath]["totalExecutionTimeMilliseconds"] = fileStats.totalExecutionTimeMilliseconds;
        outJson["fileStats"][fileStats.filePath]["totalLinearInsertionTimeMilliseconds"] = fileStats.totalLinearInsertionTimeMilliseconds;

        outJson["fileStats"][fileStats.filePath]["cacheMisses"] = fileStats.cacheMisses;
        outJson["fileStats"][fileStats.filePath]["cacheHits"] = fileStats.cacheHits;
        outJson["fileStats"][fileStats.filePath]["cacheEvictions"] = fileStats.cacheEvictions;
        outJson["fileStats"][fileStats.filePath]["cacheDirtyEvictions"] = fileStats.cacheDirtyEvictions;
        outJson["fileStats"][fileStats.filePath]["cacheInsertions"] = fileStats.cacheInsertions;

        outJson["fileStats"][fileStats.filePath]["imageInsertionCount"] = fileStats.imageInsertionCount;
        outJson["fileStats"][fileStats.filePath]["nodeBlockSplitCount"] = fileStats.nodeBlockSplitCount;
        outJson["fileStats"][fileStats.filePath]["nodeBlockReadCount"] = fileStats.nodeBlockReadCount;
        outJson["fileStats"][fileStats.filePath]["nodeBlockWriteCount"] = fileStats.nodeBlockWriteCount;
        outJson["fileStats"][fileStats.filePath]["totalReadTimeMilliseconds"] = fileStats.totalReadTimeMilliseconds;
        outJson["fileStats"][fileStats.filePath]["totalWriteTimeMilliseconds"] = fileStats.totalWriteTimeMilliseconds;
        outJson["fileStats"][fileStats.filePath]["totalSplitTimeMilliseconds"] = fileStats.totalSplitTimeMilliseconds;

        outJson["fileStats"][fileStats.filePath]["totalAllocatedImageCapacity"] = fileStats.totalAllocatedImageCapacity;
        outJson["fileStats"][fileStats.filePath]["totalCachedImageCount"] = fileStats.totalCachedImageCount;
        outJson["fileStats"][fileStats.filePath]["cachedLeafNodeCount"] = fileStats.cachedLeafNodeCount;
        outJson["fileStats"][fileStats.filePath]["cachedIntermediateNodeCount"] = fileStats.cachedIntermediateNodeCount;
        outJson["fileStats"][fileStats.filePath]["maximumImagesPerNode"] = fileStats.maximumImagesPerNode;
        outJson["fileStats"][fileStats.filePath]["maximumImagesPerNodeBlock"] = fileStats.maximumImagesPerNodeBlock;
    }

    std::ofstream outFile(path);
    outFile << outJson.dump(4);
    outFile.close();
}

Index SpinImage::index::build(
        std::experimental::filesystem::path quicciImageDumpDirectory,
        std::experimental::filesystem::path indexDumpDirectory,
        size_t cacheNodeLimit,
        size_t cacheImageLimit,
        size_t fileStartIndex,
        size_t fileEndIndex,
        bool appendToExistingIndex,
        std::experimental::filesystem::path statisticsFileDumpLocation) {
    std::vector<std::experimental::filesystem::path> filesInDirectory = SpinImage::utilities::listDirectory(quicciImageDumpDirectory);
    std::experimental::filesystem::path indexDirectory(indexDumpDirectory);
    omp_set_nested(1);

    std::vector<std::experimental::filesystem::path>* indexedFiles =
            new std::vector<std::experimental::filesystem::path>();

    if(appendToExistingIndex) {
        Index loadedIndex = SpinImage::index::io::readIndex(indexDumpDirectory);
        indexedFiles->resize(loadedIndex.indexedFileList->size());
        std::copy(loadedIndex.indexedFileList->begin(), loadedIndex.indexedFileList->end(), indexedFiles->begin());
    } else {
        indexedFiles->reserve(filesInDirectory.size());
    }

    bool enableStatisticsDump = statisticsFileDumpLocation != std::experimental::filesystem::path("/none/selected");
    if(enableStatisticsDump) {
        std::cout << "Statistics will be dumped to " << statisticsFileDumpLocation << std::endl;
    }
    std::vector<IndexedFileStatistics> fileStatistics;

    IndexConstructionSettings constructionSettings =
            {quicciImageDumpDirectory, indexDumpDirectory, cacheNodeLimit, cacheImageLimit, fileStartIndex, fileEndIndex};

    size_t endIndex = fileEndIndex == fileStartIndex ? filesInDirectory.size() : fileEndIndex;

    size_t totalImageCount = 0;
    size_t currentTotalImageIndex = 0;

        #pragma omp parallel for schedule(dynamic)
        for (unsigned int fileIndex = 0; fileIndex < endIndex; fileIndex++) {
            std::experimental::filesystem::path path = filesInDirectory.at(fileIndex);
            const std::string archivePath = path.string();

            SpinImage::cpu::QUICCIImages images = SpinImage::read::QUICCImagesFromDumpFile(archivePath);
            double totalImageDurationMilliseconds = 0;
            #pragma omp critical
            {
                totalImageCount += images.imageCount;
                indexedFiles->emplace_back(archivePath);
                std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();

                #pragma omp parallel
                {
                    std::vector<std::pair<unsigned short, unsigned short>> floodFillPixels;
                    floodFillPixels.reserve(4096);

                    #pragma omp for schedule(dynamic)
                    for (IndexImageID imageIndex = 0; imageIndex < images.imageCount; imageIndex++) {
                        /*if (imageIndex % 5000 == 0) {
                            std::stringstream progressBar;
                            progressBar << "\r[";
                            int dashCount = int((float(imageIndex) / float(images.imageCount)) * 25.0f) + 1;
                            for (int i = 0; i < 25; i++) {
                                progressBar << ((i < dashCount) ? "=" : " ");
                            }
                            progressBar << "] " << imageIndex << "/" << images.imageCount << "\r";
                            std::cout << progressBar.str() << std::flush;
                        }*/

                        std::chrono::steady_clock::time_point imageStartTime = std::chrono::steady_clock::now();
                        QuiccImage combined = combineQuiccImages(
                                images.horizontallyIncreasingImages[imageIndex],
                                images.horizontallyDecreasingImages[imageIndex]);
                        IndexEntry entry = {fileIndex, imageIndex};


                        for (unsigned int row = 0; row < 64; row++) {
                            for (unsigned int col = 0; col < 64; col++) {

                                unsigned int pixel = (unsigned int) ((
                                                         combined.at(2 * row + (col / 32))
                                                                 >> (31U - col)) & 0x1U);

                                if (pixel == 1) {
                                    unsigned int regionSize = 0;
                                    floodFillPixels.clear();
                                    floodFillPixels.emplace_back(row, col);
                                    std::fill(patternImage.begin(), patternImage.end(), 0);

                                    while (!floodFillPixels.empty()) {
                                        std::pair<unsigned short, unsigned short> pixelIndex = floodFillPixels.at(
                                                floodFillPixels.size() - 1);
                                        floodFillPixels.erase(floodFillPixels.begin() + floodFillPixels.size() - 1);
                                        unsigned int chunkIndex = 2 * pixelIndex.first + (pixelIndex.second / 32);
                                        unsigned int chunk = combined.at(chunkIndex);
                                        unsigned int floodPixel = (unsigned int)
                                                ((chunk >> (31U - pixelIndex.second % 32)) & 0x1U);
                                        if (floodPixel == 1) {
                                            regionSize++;
                                            // Add pixel to pattern image
                                            unsigned int bitEnablingMask = 0x1U << (31U - pixelIndex.second % 32);
                                            patternImage.at(chunkIndex) |= bitEnablingMask;
                                            // Disable pixel
                                            unsigned int bitDisablingMask = ~bitEnablingMask;
                                            combined.at(chunkIndex) = chunk & bitDisablingMask;
                                            // Queue surrounding pixels
                                            const int range = 2;
                                            for (int floodRow = std::max(int(pixelIndex.first) - range, 0);
                                                     floodRow <= std::min(63, pixelIndex.first + range);
                                                     floodRow++) {
                                                for (int floodCol = std::max(int(pixelIndex.second) - range, 0);
                                                         floodCol <= std::min(63, pixelIndex.second + range);
                                                         floodCol++) {
                                                    floodFillPixels.emplace_back(floodRow, floodCol);
                                                }
                                            }
                                        }
                                    }

                                    if(regionSize-1 >= minSize && regionSize-1 < maxSize && seenPatterns.at(regionSize-1).find(patternImage) == seenPatterns.at(regionSize-1).end()) {
                                        threadSeenPatterns.at(regionSize - 1).insert(patternImage);
                                    }

                                }
                            }
                        }

                        std::chrono::steady_clock::time_point imageEndTime = std::chrono::steady_clock::now();
                        #pragma omp atomic
                        totalImageDurationMilliseconds += std::chrono::duration_cast<std::chrono::nanoseconds>(
                                imageEndTime - imageStartTime).count() / 1000000.0;
                    }
                }

                std::chrono::steady_clock::time_point endTime = std::chrono::steady_clock::now();
                double durationMilliseconds =
                        std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count() / 1000000.0;

                if (enableStatisticsDump && fileIndex % 100 == 99) {
                    std::cout << "Writing statistics file..                                      \n";
                    dumpStatisticsFile(fileStatistics, constructionSettings, statisticsFileDumpLocation);
                }

                std::cout << "Added file " << (fileIndex + 1) << "/" << endIndex
                          << ": " << archivePath
                          << ", Duration: " << (durationMilliseconds / 1000.0) << "s"
                          << ", Image count: " << images.imageCount << std::endl;
            };

            // Necessity to prevent libc from hogging all system memory
            malloc_trim(0);

            delete[] images.horizontallyIncreasingImages;
            delete[] images.horizontallyDecreasingImages;
        }

        malloc_trim(0);


    std::cout << std::endl << "Total Added Image Count: " << totalImageCount << std::endl;

    dumpStatisticsFile(fileStatistics, constructionSettings, statisticsFileDumpLocation);

    // Final construction of the index
    Index index(indexDirectory, indexedFiles);

    // Write the root node to disk
    std::cout << "Writing core index files.." << std::endl;
    SpinImage::index::io::writeIndex(index, indexDirectory);

    return index;
}


