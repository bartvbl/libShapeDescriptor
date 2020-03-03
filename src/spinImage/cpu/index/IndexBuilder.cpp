#include <spinImage/utilities/fileutils.h>
#include <iostream>
#include <spinImage/cpu/types/QUICCIImages.h>
#include <spinImage/utilities/readers/quicciReader.h>
#include <spinImage/cpu/index/types/MipmapStack.h>
#include <bitset>
#include <spinImage/cpu/types/QuiccImage.h>
#include <json.hpp>
#include <fstream>
#include "IndexBuilder.h"
#include "NodeBlockCache.h"
#include "tsl/ordered_map.h"

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
    size_t cachedItemCount;
    double totalExecutionTimeMilliseconds;

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
};

struct IndexConstructionSettings {
    std::experimental::filesystem::path quicciImageDumpDirectory;
    std::experimental::filesystem::path indexDumpDirectory;
    size_t cacheCapacity;
};

IndexedFileStatistics gatherFileStatistics(
        const NodeBlockCache *cache,
        size_t fileIndex,
        double totalExecutionTimeMilliseconds,
        size_t imageCount,
        const std::string &filePath) {
    IndexedFileStatistics stats;
    stats.filePath = filePath;
    stats.fileIndex = fileIndex;
    stats.imageCount = imageCount;
    stats.cachedItemCount = cache->getCurrentItemCount();
    stats.totalExecutionTimeMilliseconds = totalExecutionTimeMilliseconds;

    stats.cacheMisses = cache->statistics.misses;
    stats.cacheHits = cache->statistics.hits;
    stats.cacheEvictions = cache->statistics.evictions;
    stats.cacheDirtyEvictions = cache->statistics.dirtyEvictions;
    stats.cacheInsertions = cache->statistics.insertions;

    stats.imageInsertionCount = cache->nodeBlockStatistics.imageInsertionCount;
    stats.nodeBlockSplitCount = cache->nodeBlockStatistics.nodeSplitCount;
    stats.nodeBlockReadCount = cache->nodeBlockStatistics.totalReadCount;
    stats.nodeBlockWriteCount = cache->nodeBlockStatistics.totalWriteCount;
    stats.totalReadTimeMilliseconds = cache->nodeBlockStatistics.totalReadTimeMilliseconds;
    stats.totalWriteTimeMilliseconds = cache->nodeBlockStatistics.totalWriteTimeMilliseconds;
    stats.totalSplitTimeMilliseconds = cache->nodeBlockStatistics.totalSplitTimeMilliseconds;

    return stats;
}

void dumpStatisticsFile(
        const std::vector<IndexedFileStatistics> &fileStatistics,
        const IndexConstructionSettings &constructionSettings,
        const std::experimental::filesystem::path &path) {
    json outJson;

    outJson["version"] = "v1";
    outJson["nodesPerBlock"] = NODES_PER_BLOCK;
    outJson["nodeSplitThreshold"] = NODE_SPLIT_THRESHOLD;
    outJson["cacheCapacity"] = constructionSettings.cacheCapacity;
    outJson["outputIndexDirectory"] = constructionSettings.indexDumpDirectory;
    outJson["inputImageDirectory"] = constructionSettings.quicciImageDumpDirectory;

    outJson["fileStats"] = {};
    for(const IndexedFileStatistics &fileStats : fileStatistics) {
        outJson["fileStats"][fileStats.filePath] = {};

        outJson["fileStats"][fileStats.filePath]["fileIndex"] = fileStats.fileIndex;
        outJson["fileStats"][fileStats.filePath]["imageCount"] = fileStats.imageCount;
        outJson["fileStats"][fileStats.filePath]["cachedItemCount"] = fileStats.cachedItemCount;
        outJson["fileStats"][fileStats.filePath]["totalExecutionTimeMilliseconds"] = fileStats.totalExecutionTimeMilliseconds;

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
    }

    std::ofstream outFile(path);
    outFile << outJson.dump(4);
    outFile.close();
}

Index SpinImage::index::build(
        std::experimental::filesystem::path quicciImageDumpDirectory,
        std::experimental::filesystem::path indexDumpDirectory,
        std::experimental::filesystem::path statisticsFileDumpLocation) {
    std::vector<std::experimental::filesystem::path> filesInDirectory = SpinImage::utilities::listDirectory(quicciImageDumpDirectory);
    std::experimental::filesystem::path indexDirectory(indexDumpDirectory);
    omp_set_nested(1);

    std::vector<std::experimental::filesystem::path>* indexedFiles =
            new std::vector<std::experimental::filesystem::path>();
    bool enableStatisticsDump = statisticsFileDumpLocation == std::experimental::filesystem::path("/none/selected");
    indexedFiles->reserve(filesInDirectory.size());
    std::vector<IndexedFileStatistics> fileStatistics;

    NodeBlock rootBlock;

    const size_t cacheCapacity = 50000;
    NodeBlockCache cache(cacheCapacity, indexDirectory, &rootBlock);

    IndexConstructionSettings constructionSettings =
            {quicciImageDumpDirectory, indexDumpDirectory, cacheCapacity};

#pragma omp parallel for schedule(dynamic)
    for(unsigned int fileIndex = 0; fileIndex < 2500 /*filesInDirectory.size()*/; fileIndex++) {
        std::experimental::filesystem::path path = filesInDirectory.at(fileIndex);
        const std::string archivePath = path.string();

        SpinImage::cpu::QUICCIImages images = SpinImage::read::QUICCImagesFromDumpFile(archivePath);

        std::chrono::duration<double, std::milli> duration;

#pragma omp critical
        {
            indexedFiles->emplace_back(archivePath);
            std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();
            for (IndexImageID imageIndex = 0; imageIndex < images.imageCount; imageIndex++) {
                QuiccImage combined = MipmapStack::combine(
                        images.horizontallyIncreasingImages[imageIndex],
                        images.horizontallyDecreasingImages[imageIndex]);
                IndexEntry entry = {fileIndex, imageIndex};

                cache.insertImage(combined, entry);
            }

            std::chrono::steady_clock::time_point endTime = std::chrono::steady_clock::now();
            duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

            std::cout << std::endl
                      << "Added file " << (fileIndex + 1) << "/" << filesInDirectory.size()
                      << ": " << archivePath
                      << ", Cache: " << cache.getCurrentItemCount() << "/" << cache.itemCapacity
                      << ", Duration: " << (duration.count() / 1000.0) << "s" << std::endl;
        };

        delete[] images.horizontallyIncreasingImages;
        delete[] images.horizontallyDecreasingImages;

        if(enableStatisticsDump) {
#pragma omp critical
            {
                fileStatistics.push_back(
                        gatherFileStatistics(&cache, fileIndex, duration.count(), images.imageCount, archivePath));
                cache.statistics.reset();
                cache.nodeBlockStatistics.reset();
                dumpStatisticsFile(fileStatistics, constructionSettings, statisticsFileDumpLocation);
            };
        }
    }

    // Ensuring all changes are written to disk
    std::cout << "Flushing cache.." << std::endl;
    cache.flush();

    dumpStatisticsFile(fileStatistics, constructionSettings, statisticsFileDumpLocation);

    // Final construction of the index
    Index index(indexDirectory, indexedFiles, rootBlock);

    // Write the root node to disk
    std::cout << "Writing core index files.." << std::endl;
    SpinImage::index::io::writeNodeBlock(&rootBlock, indexDirectory);
    SpinImage::index::io::writeIndex(index, indexDirectory);

    return index;
}