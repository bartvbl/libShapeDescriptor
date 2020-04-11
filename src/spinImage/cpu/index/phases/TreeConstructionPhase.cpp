#include <fast-lzma2.h>
#include <spinImage/cpu/index/phases/types/FileEntry.h>
#include <fstream>
#include <cassert>
#include <mutex>
#include <iostream>
#include <set>
#include <spinImage/cpu/index/Pattern.h>
#include <spinImage/utilities/compression/FileDecompressionStream.h>
#include "TreeConstructionPhase.h"

void constructIndexTree(std::experimental::filesystem::path quicciImageDumpDirectory,
                        std::experimental::filesystem::path indexDumpDirectory,
                        size_t cachedPatternLimit,
                        size_t fileStartIndex, size_t fileEndIndex) {

    std::array<size_t, 4096> totalLooseNodeCount;
    std::fill(totalLooseNodeCount.begin(), totalLooseNodeCount.end(), 0);

    std::set<QuiccImage> currentLayerContents;
    std::set<QuiccImage> previousLayerContents;
    for(int patternSize = 1; patternSize <= 4096; patternSize++) {
        FL2_DStream* decompressionStream = FL2_createDStream();
        FL2_initDStream(decompressionStream);

        std::experimental::filesystem::path patternStatisticsDirectory(indexDumpDirectory / "patternStats");
        std::experimental::filesystem::path dumpFileLocation =
                patternStatisticsDirectory / ("pattern_stats_" + std::to_string(patternSize) + ".dat");

        std::fstream patternFileStream(dumpFileLocation, std::ios::binary | std::ios::in);

        // Sanity check: make sure the file ID bytes are correct
        char fileHeader[4];
        patternFileStream.read(fileHeader, 4);
        assert(std::string(fileHeader) == "PCF");

        // Buffer sizes
        size_t fileEntryCount;
        patternFileStream.read(reinterpret_cast<char *>(&fileEntryCount), sizeof(size_t));
        size_t compressedFileEntryListSize;
        patternFileStream.read(reinterpret_cast<char *>(&compressedFileEntryListSize), sizeof(size_t));

        std::mutex streamLock;
        std::mutex setLock;

        SpinImage::utilities::FileDecompressionStream<FileEntry, 8> inStream(&patternFileStream,
                compressedFileEntryListSize, fileEntryCount);

        //#pragma omp parallel for
        while(!inStream.isDepleted()) {
            streamLock.lock();
            std::array<FileEntry, 8> uncompressedBuffer;
            unsigned int bufferImageCount = inStream.read(uncompressedBuffer);
            streamLock.unlock();

            // Copy images into set
            setLock.lock();
            for(int i = 0; i < bufferImageCount; i++) {
                currentLayerContents.insert(uncompressedBuffer.at(i).image);
            }
            setLock.unlock();

            if(patternSize > 1) {
                // Attempt to locate images in previous layer
                // Can only be done if said previous layer exists
                for(int i = 0; i < bufferImageCount; i++) {
                    QuiccImage patternImage = uncompressedBuffer.at(i).image;
                    bool parentFound = false;
                    for(int row = 0; row < spinImageWidthPixels && !parentFound; row++) {
                        for(int col = 0; col < spinImageWidthPixels && !parentFound; col++) {
                            unsigned int chunkIndex = SpinImage::index::pattern::computeChunkIndex(row, col);
                            unsigned int pixel = SpinImage::index::pattern::pixelAt(patternImage, row, col);
                            if(pixel == 1) {
                                // Disable pixel
                                unsigned int bitEnableMask = SpinImage::index::pattern::computeBitMask(col);
                                patternImage.at(chunkIndex) &= ~bitEnableMask;
                                // Perform lookup
                                bool parentExists = previousLayerContents.find(patternImage) != previousLayerContents.end();
                                parentFound = parentExists;
                                // Re-enable pixel
                                patternImage.at(chunkIndex) |= bitEnableMask;
                            }
                        }
                    }
                    if(!parentFound) {
                        #pragma omp atomic
                        totalLooseNodeCount.at(patternSize-1)++;
                    }
                }
            }
        }
        
        if(patternSize > 1) {
            size_t nodesWithParentCount = currentLayerContents.size() - totalLooseNodeCount.at(patternSize-1);
            std::cout << "(" << patternSize << ", " << nodesWithParentCount << ", " << currentLayerContents.size() << ", " << totalLooseNodeCount.at(patternSize-1) << "), " << std::flush;

            previousLayerContents.clear();
        }
        previousLayerContents.swap(currentLayerContents);
    }



}