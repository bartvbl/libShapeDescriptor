#include <spinImage/utilities/fileutils.h>
#include <iostream>
#include <bitset>
#include <spinImage/cpu/types/QuiccImage.h>
#include <json.hpp>
#include "IndexBuilder.h"
#include "tsl/ordered_map.h"
#include "IndexIO.h"
#include "Pattern.h"
#include <fast-lzma2.h>
#include <omp.h>
#include <spinImage/cpu/index/listConstructor/ListConstructor.h>

template<class Key, class T, class Ignore, class Allocator,
        class Hash = std::hash<Key>, class KeyEqual = std::equal_to<Key>,
        class AllocatorPair = typename std::allocator_traits<Allocator>::template rebind_alloc<std::pair<Key, T>>,
        class ValueTypeContainer = std::vector<std::pair<Key, T>, AllocatorPair>>
using ordered_map = tsl::ordered_map<Key, T, Hash, KeyEqual, AllocatorPair, ValueTypeContainer>;

using json = nlohmann::basic_json<ordered_map>;

struct IndexConstructionSettings {
    std::experimental::filesystem::path quicciImageDumpDirectory;
    std::experimental::filesystem::path indexDumpDirectory;
    size_t cacheNodeBlockCapacity;
    size_t cacheImageCapacity;
    size_t fileStartIndex;
    size_t fileEndIndex;
};

void dumpStatisticsFile(
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

    std::ofstream outFile(path);
    outFile << outJson.dump(4);
    outFile.close();
}

void SpinImage::index::build(
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

    IndexConstructionSettings constructionSettings =
            {quicciImageDumpDirectory, indexDumpDirectory, cacheNodeLimit, cacheImageLimit, fileStartIndex, fileEndIndex};

    size_t endIndex = fileEndIndex == fileStartIndex ? filesInDirectory.size() : fileEndIndex;

    // Phase 1:
    buildInitialPixelLists(quicciImageDumpDirectory, indexDumpDirectory, cacheNodeLimit, fileStartIndex, endIndex);

    dumpStatisticsFile(constructionSettings, statisticsFileDumpLocation);
}


