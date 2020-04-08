#include <fast-lzma2.h>
#include <spinImage/cpu/index/phases/types/FileEntry.h>
#include <fstream>
#include <cassert>
#include <mutex>
#include <iostream>
#include <set>
#include "TreeConstructionPhase.h"

void constructIndexTree(std::experimental::filesystem::path quicciImageDumpDirectory,
                        std::experimental::filesystem::path indexDumpDirectory,
                        size_t cachedPatternLimit,
                        size_t fileStartIndex, size_t fileEndIndex) {

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

        const unsigned int imagesPerBuffer = 8;
        const size_t bufferSize = imagesPerBuffer * sizeof(FileEntry);
        unsigned char cBuffer[bufferSize];
        FL2_inBuffer compressedBuffer = {cBuffer, bufferSize, bufferSize};
        size_t totalReadCompressedBytes = 0;
        size_t totalUncompressedBytes = 0;

        std::mutex streamLock;

        unsigned int blockCount = fileEntryCount % imagesPerBuffer == 0 ?
                                  (fileEntryCount / imagesPerBuffer) :
                                  (fileEntryCount / imagesPerBuffer) + 1;

        unsigned int blockIndex = 0;

        #pragma omp parallel for
        for(unsigned int blockIterationIndex = 0; blockIterationIndex < blockCount; blockIterationIndex++) {
            streamLock.lock();
            unsigned int acquiredBlockIndex = 0;

            FileEntry uBuffer[imagesPerBuffer];
            FL2_outBuffer uncompressedBuffer = {uBuffer, imagesPerBuffer * sizeof(FileEntry), 0};
            unsigned int decompressedBytesThisIteration = 0;

            while(uncompressedBuffer.pos < bufferSize && totalUncompressedBytes < fileEntryCount * sizeof(FileEntry)) {
                // Refill the compressed buffer
                if (compressedBuffer.pos == compressedBuffer.size) {
                    size_t numberOfBytesToRead =
                            std::min<size_t>(
                                    (compressedFileEntryListSize * sizeof(FileEntry)) - totalReadCompressedBytes,
                                    bufferSize);
                    compressedBuffer.size = numberOfBytesToRead;
                    patternFileStream.read((char*) cBuffer, numberOfBytesToRead);
                    compressedBuffer.pos = 0;

                    totalReadCompressedBytes += numberOfBytesToRead;;
                }
                FL2_decompressStream(decompressionStream, &uncompressedBuffer, &compressedBuffer);
                decompressedBytesThisIteration = uncompressedBuffer.pos - decompressedBytesThisIteration;
                totalUncompressedBytes += decompressedBytesThisIteration;
            }

            acquiredBlockIndex = blockIndex;
            blockIndex++;

            streamLock.unlock();

            // Ensure uncompressed buffer is as full as it can be
            assert(uncompressedBuffer.pos == imagesPerBuffer * sizeof(FileEntry)
               || (acquiredBlockIndex + 1 == blockCount
                   && uncompressedBuffer.pos == ((fileEntryCount % imagesPerBuffer) * sizeof(FileEntry))));

            // Ensure contents of uncompressed buffer contain complete images
            assert(uncompressedBuffer.pos % sizeof(FileEntry) == 0);
            unsigned int bufferImageCount = uncompressedBuffer.pos / sizeof(FileEntry);

            // Copy images into complete array
            for(int i = 0; i < bufferImageCount; i++) {
                currentLayerContents.insert(uBuffer[i].image);
            }

            
        }

        FL2_freeDStream(decompressionStream);

        if(patternSize > 1) {
            previousLayerContents.clear();
        }
        previousLayerContents.swap(currentLayerContents);
    }



}