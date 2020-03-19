#include <spinImage/utilities/fileutils.h>
#include <iostream>
#include <spinImage/cpu/types/QUICCIImages.h>
#include <spinImage/utilities/readers/quicciReader.h>
#include <bitset>
#include <spinImage/cpu/types/QuiccImage.h>
#include <json.hpp>
#include <fstream>
#include "IndexBuilder.h"
#include "NodeBlockCache.h"
#include "tsl/ordered_map.h"

#include <fast-lzma2.h>
#include <malloc.h>

template<class Key, class T, class Ignore, class Allocator,
        class Hash = std::hash<Key>, class KeyEqual = std::equal_to<Key>,
        class AllocatorPair = typename std::allocator_traits<Allocator>::template rebind_alloc<std::pair<Key, T>>,
        class ValueTypeContainer = std::vector<std::pair<Key, T>, AllocatorPair>>
using ordered_map = tsl::ordered_map<Key, T, Hash, KeyEqual, AllocatorPair, ValueTypeContainer>;

using json = nlohmann::basic_json<ordered_map>;


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
        NodeBlockCache *cache,
        size_t fileIndex,
        double totalImageDurationMilliseconds,
        double totalExecutionTimeMilliseconds,
        size_t imageCount,
        const std::string &filePath) {
    /*unsigned long totalCapacity = 0;
    unsigned long totalImageCount = 0;
    unsigned long maximumImageCount = 0;
    unsigned long leafNodeCount = 0;
    unsigned long intermediateNodeCount = 0;
    unsigned long maximumImagesPerNode = 0;
    for(CachedItem<std::string, NodeBlock> &block : cache->lruItemQueue) {
        unsigned int entryCount = 0;
        unsigned long nodeImageCount = 0;
        for(int i = 0; i < NODES_PER_BLOCK; i++) {
            const auto& entry = block.item->leafNodeContents.at(i);
            entryCount += entry.capacity();
            totalImageCount += entry.size();
            nodeImageCount += entry.size();
            maximumImageCount = std::max<unsigned long>(maximumImageCount, entry.size());
            if(block.item->childNodeIsLeafNode[i]) {
                leafNodeCount++;
            } else {
                intermediateNodeCount++;
            }
        }
        maximumImagesPerNode = std::max<unsigned long>(maximumImagesPerNode, nodeImageCount);
        totalCapacity += entryCount;
    }
    std::cout << (double(totalImageCount*sizeof(NodeBlockEntry)) / double(1024*1024*1024)) << "GB/" << (double(totalCapacity*sizeof(NodeBlockEntry)) / double(1024*1024*1024)) << "GB (max " << maximumImageCount << ", max per node " << maximumImagesPerNode << ", average " << (double(totalCapacity) / double(leafNodeCount)) << ")" << std::endl;*/

    IndexedFileStatistics stats;
    stats.filePath = filePath;
    stats.fileIndex = fileIndex;
    stats.imageCount = imageCount;
    stats.cachedNodeBlockCount = cache->getCurrentItemCount();
    stats.cachedImageCount = cache->getCurrentImageCount();
    stats.totalExecutionTimeMilliseconds = totalExecutionTimeMilliseconds;
    stats.totalLinearInsertionTimeMilliseconds = totalImageDurationMilliseconds;

    stats.cacheMisses = cache->statistics.misses;
    stats.cacheHits = cache->statistics.hits;
    stats.cacheEvictions = cache->statistics.evictions;
    stats.cacheDirtyEvictions = cache->statistics.dirtyEvictions;
    stats.cacheInsertions = cache->statistics.insertions;

    stats.imageInsertionCount = cache->nodeBlockStatistics.imageInsertionCount;
    stats.nodeBlockSplitCount = cache->nodeBlockStatistics.nodeSplitCount;
    stats.nodeBlockReadCount = cache->nodeBlockStatistics.totalReadCount;
    stats.nodeBlockWriteCount = cache->nodeBlockStatistics.totalWriteCount;
    stats.totalReadTimeMilliseconds = cache->nodeBlockStatistics.totalReadTimeNanoseconds / 1000000.0;
    stats.totalWriteTimeMilliseconds = cache->nodeBlockStatistics.totalWriteTimeNanoseconds / 1000000.0;
    stats.totalSplitTimeMilliseconds = cache->nodeBlockStatistics.totalSplitTimeNanoseconds / 1000000.0;

    /*stats.totalAllocatedImageCapacity = totalCapacity;
    stats.totalCachedImageCount = totalImageCount;
    stats.cachedLeafNodeCount = leafNodeCount;
    stats.cachedIntermediateNodeCount = intermediateNodeCount;
    stats.maximumImagesPerNode = maximumImageCount;
    stats.maximumImagesPerNodeBlock = maximumImagesPerNode;*/

    return stats;
}

void dumpStatisticsFile(
        const std::vector<IndexedFileStatistics> &fileStatistics,
        const IndexConstructionSettings &constructionSettings,
        const std::experimental::filesystem::path &path) {
    json outJson;

    outJson["version"] = "v4";
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
    std::cout << "Sizes: " << sizeof(NodeBlock) << ", " << sizeof(NodeBlockEntry) << std::endl;
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

    NodeBlockCache cache(cacheNodeLimit, cacheImageLimit, indexDirectory, appendToExistingIndex);

    IndexConstructionSettings constructionSettings =
            {quicciImageDumpDirectory, indexDumpDirectory, cacheNodeLimit, cacheImageLimit, fileStartIndex, fileEndIndex};

    size_t endIndex = fileEndIndex == fileStartIndex ? filesInDirectory.size() : fileEndIndex;

    #pragma omp parallel for schedule(dynamic)
    for(unsigned int fileIndex = 0; fileIndex < endIndex; fileIndex++) {
        std::experimental::filesystem::path path = filesInDirectory.at(fileIndex);
        const std::string archivePath = path.string();

        SpinImage::cpu::QUICCIImages images = SpinImage::read::QUICCImagesFromDumpFile(archivePath);
        double totalImageDurationMilliseconds = 0;
        #pragma omp critical
        {
            indexedFiles->emplace_back(archivePath);
            std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();

            #pragma omp parallel for schedule(dynamic)
            for (IndexImageID imageIndex = 0; imageIndex < images.imageCount; imageIndex++) {
                std::chrono::steady_clock::time_point imageStartTime = std::chrono::steady_clock::now();
                QuiccImage combined = combineQuiccImages(
                        images.horizontallyIncreasingImages[imageIndex],
                        images.horizontallyDecreasingImages[imageIndex]);
                IndexEntry entry = {fileIndex, imageIndex};
                cache.insertImage(combined, entry);
                std::chrono::steady_clock::time_point imageEndTime = std::chrono::steady_clock::now();
                #pragma omp atomic
                totalImageDurationMilliseconds += std::chrono::duration_cast<std::chrono::nanoseconds>(imageEndTime - imageStartTime).count() / 1000000.0;
                // Occasionally force frees memory such that memory does not accumulate too much over time
                if(imageIndex % 25000 == 0) {
                    malloc_trim(0);
                }
            }

            std::chrono::steady_clock::time_point endTime = std::chrono::steady_clock::now();
            double durationMilliseconds =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count() / 1000000.0;

            fileStatistics.push_back(gatherFileStatistics(&cache, fileIndex, totalImageDurationMilliseconds, durationMilliseconds, images.imageCount, archivePath));

            cache.statistics.reset();
            cache.nodeBlockStatistics.reset();

            if(enableStatisticsDump && fileIndex % 100 == 99) {
                std::cout << "Writing statistics file..\n";
                dumpStatisticsFile(fileStatistics, constructionSettings, statisticsFileDumpLocation);
            }

            std::cout << "Added file " << (fileIndex + 1) << "/" << endIndex
                      << ": " << archivePath
                      << ", Cache (nodes: " << cache.getCurrentItemCount() << "/" << cache.itemCapacity
                      << ", images: " << cache.getCurrentImageCount() << "/" << cache.imageCapacity << ")"
                      << ", Duration: " << (durationMilliseconds / 1000.0) << "s"
                      << ", Image count: " << images.imageCount << std::endl;
        };

        // Necessity to prevent libc from hogging all system memory
        malloc_trim(0);

        delete[] images.horizontallyIncreasingImages;
        delete[] images.horizontallyDecreasingImages;
    }

    // Ensuring all changes are written to disk
    std::cout << "Flushing cache.." << std::endl;
    cache.flush();

    dumpStatisticsFile(fileStatistics, constructionSettings, statisticsFileDumpLocation);

    // Final construction of the index
    Index index(indexDirectory, indexedFiles);

    // Write the root node to disk
    std::cout << "Writing core index files.." << std::endl;
    SpinImage::index::io::writeIndex(index, indexDirectory);

    return index;
}
