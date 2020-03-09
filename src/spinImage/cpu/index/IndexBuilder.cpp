#include <spinImage/utilities/fileutils.h>
#include <iostream>
#include <spinImage/cpu/types/QUICCIImages.h>
#include <spinImage/utilities/readers/quicciReader.h>
#include <spinImage/cpu/index/types/MipmapStack.h>
#include <bitset>
#include <spinImage/cpu/types/QuiccImage.h>
#include <json.hpp>
#include <fstream>
#include <spinImage/cpu/index/types/CannonFodderCache.h>
#include "IndexBuilder.h"
#include "NodeBlockCache.h"
#include "tsl/ordered_map.h"

#include <fast-lzma2.h>

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
};

struct IndexConstructionSettings {
    std::experimental::filesystem::path quicciImageDumpDirectory;
    std::experimental::filesystem::path indexDumpDirectory;
    size_t cacheNodeBlockCapacity;
    size_t cacheImageCapacity;
};

IndexedFileStatistics gatherFileStatistics(
        const NodeBlockCache *cache,
        size_t fileIndex,
        double totalImageDurationMilliseconds,
        double totalExecutionTimeMilliseconds,
        size_t imageCount,
        const std::string &filePath) {
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

    return stats;
}

void dumpStatisticsFile(
        const std::vector<IndexedFileStatistics> &fileStatistics,
        const IndexConstructionSettings &constructionSettings,
        const std::experimental::filesystem::path &path) {
    json outJson;

    outJson["version"] = "v2";
    outJson["nodesPerBlock"] = NODES_PER_BLOCK;
    outJson["nodeSplitThreshold"] = NODE_SPLIT_THRESHOLD;
    outJson["cacheNodeBlockCapacity"] = constructionSettings.cacheNodeBlockCapacity;
    outJson["cacheImageCapacity"] = constructionSettings.cacheImageCapacity;
    outJson["outputIndexDirectory"] = constructionSettings.indexDumpDirectory;
    outJson["inputImageDirectory"] = constructionSettings.quicciImageDumpDirectory;

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
    std::cout << "Sizes: " << sizeof(NodeBlock) << ", " << sizeof(NodeBlockEntry) << std::endl;
    std::experimental::filesystem::path indexDirectory(indexDumpDirectory);
    omp_set_nested(1);

    std::vector<std::experimental::filesystem::path>* indexedFiles =
            new std::vector<std::experimental::filesystem::path>();
    bool enableStatisticsDump = statisticsFileDumpLocation != std::experimental::filesystem::path("/none/selected");
    if(enableStatisticsDump) {
        std::cout << "Statistics will be dumped to " << statisticsFileDumpLocation << std::endl;
    }
    indexedFiles->reserve(filesInDirectory.size());
    std::vector<IndexedFileStatistics> fileStatistics;

    const size_t cacheNodeBlockCapacity = 1000;
    const size_t cacheImageCapacity = 500000;
    NodeBlockCache cache(cacheNodeBlockCapacity, cacheImageCapacity, indexDirectory);

    IndexConstructionSettings constructionSettings =
            {quicciImageDumpDirectory, indexDumpDirectory, cacheNodeBlockCapacity, cacheImageCapacity};

/*for(int i = 0; i < 100; i++) {
    std::cout << "lol " << i << std::endl;
#pragma omp parallel for
    for(unsigned int fileIndex = 0; fileIndex < 75 filesInDirectory.size(); fileIndex++) {
        std::cout << "hai " + std::to_string(fileIndex) + "\n";
        std::experimental::filesystem::path path = filesInDirectory.at(fileIndex);
        const std::string archivePath = path.string();

        SpinImage::cpu::QUICCIImages images = SpinImage::read::QUICCImagesFromDumpFile(archivePath);

        delete[] images.horizontallyIncreasingImages;
        delete[] images.horizontallyDecreasingImages;
    }
}*/
 /*   SpinImage::cpu::QUICCIImages tempImages = SpinImage::read::QUICCImagesFromDumpFile(filesInDirectory.at(0));
    std::cout << "hello " << tempImages.imageCount << std::endl;
    CannonFodderCache<NodeBlock> lolCache(500);
    for(size_t i = 0; i < 10000000; i++) {
        std::cout << "lol " << i << std::endl;
#pragma omp parallel for
        for(unsigned int fileIndex = 0; fileIndex < 250; fileIndex++) {
            NodeBlock* block = new NodeBlock();

            for(int j = 0; j < tempImages.imageCount; j++) {
                block->leafNodeContents.at(j % 256).emplace_back(IndexEntry(0, 0), tempImages.horizontallyDecreasingImages[j]);
            }
            lolCache.insertSomeBlock(i, block);
        }
    }
    lolCache.flush();
*/
  /*  for(size_t i = 0; i < 10000000; i++) {
        std::cout << "lol " << i << std::endl;
#pragma omp parallel
        {
            size_t bufferSize = 512 * sizeof(QuiccImage);
            QuiccImage *imageArray = new QuiccImage[512];
            char* compressedImages = new char[bufferSize];
            for (unsigned int fileIndex = 0; fileIndex < 1000; fileIndex++) {
                FL2_compress(
                        (void*) compressedImages, bufferSize,
                        (void*) imageArray, bufferSize,
                        9);
                FL2_decompress(
                        (void*) imageArray, bufferSize,
                        (void*) compressedImages, bufferSize);
            }
            delete[] imageArray;
            delete[] compressedImages;
        }
    }*/

    #pragma omp parallel for schedule(dynamic)
    for(unsigned int fileIndex = 0; fileIndex < 75 /*filesInDirectory.size()*/; fileIndex++) {
        std::experimental::filesystem::path path = filesInDirectory.at(fileIndex);
        const std::string archivePath = path.string();

        SpinImage::cpu::QUICCIImages images = SpinImage::read::QUICCImagesFromDumpFile(archivePath);
        double totalImageDurationMilliseconds = 0;
        #pragma omp critical
        {
            indexedFiles->emplace_back(archivePath);
            std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();
            #pragma omp parallel for schedule(dynamic)
            for (IndexImageID imageIndex = 0; imageIndex < /*std::min<unsigned int>(*/images.imageCount/*, 20000)*/; imageIndex++) {
                std::chrono::steady_clock::time_point imageStartTime = std::chrono::steady_clock::now();
                QuiccImage combined = MipmapStack::combine(
                        images.horizontallyIncreasingImages[imageIndex],
                        images.horizontallyDecreasingImages[imageIndex]);
                IndexEntry entry = {fileIndex, imageIndex};
                //std::cout << "Thread " + std::to_string(omp_get_thread_num()) + " is inserting image " + std::to_string(imageIndex)  + "\n";
                cache.insertImage(combined, entry);
                std::chrono::steady_clock::time_point imageEndTime = std::chrono::steady_clock::now();
                #pragma omp atomic
                totalImageDurationMilliseconds += std::chrono::duration_cast<std::chrono::nanoseconds>(imageEndTime - imageStartTime).count() / 1000000.0;
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
            //if(fileIndex % 10 == 9) {
            //    cache.flush();
            //}

            std::cout << "Added file " << (fileIndex + 1) << "/" << filesInDirectory.size()
                      << ": " << archivePath
                      << ", Cache (nodes: " << cache.getCurrentItemCount() << "/" << cache.itemCapacity
                      << ", images: " << cache.getCurrentImageCount() << "/" << cache.imageCapacity << ")"
                      << ", Duration: " << (durationMilliseconds / 1000.0) << "s"
                      << ", Image count: " << images.imageCount << std::endl;

            /*unsigned int totalImageCount = 0;
            for(CachedItem<std::string, NodeBlock> &block : cache.lruItemQueue) {
                unsigned int entryCount = 0;
                for(const auto& entry : block.item->leafNodeContents) {
                    entryCount += entry.size();
                }
                totalImageCount += entryCount;
            }
            std::cout << totalImageCount << " vs " << cache.getCurrentImageCount() << std::endl;
            assert(totalImageCount == cache.getCurrentImageCount());*/
            unsigned long totalCapacity = 0;
            for(CachedItem<std::string, NodeBlock> &block : cache.lruItemQueue) {
                unsigned int entryCount = 0;
                for(const auto& entry : block.item->leafNodeContents) {
                    entryCount += entry.capacity();
                }
                totalCapacity += entryCount;
            }
            std::cout << totalCapacity << std::endl;
        };


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
