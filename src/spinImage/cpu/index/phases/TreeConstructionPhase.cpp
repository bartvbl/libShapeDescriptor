#include <fast-lzma2.h>
#include <spinImage/cpu/index/phases/types/FileEntry.h>
#include <fstream>
#include <cassert>
#include <mutex>
#include <iostream>
#include <set>
#include <spinImage/cpu/index/Pattern.h>
#include <spinImage/utilities/compression/FileDecompressionStream.h>
#include <omp.h>
#include <bitset>
#include "TreeConstructionPhase.h"

std::fstream openStatisticsFile(const std::experimental::filesystem::path &indexDumpDirectory, int patternSize) {
    std::experimental::filesystem::path patternStatisticsDirectory(indexDumpDirectory / "patternStats");
    std::experimental::filesystem::path dumpFileLocation =
            patternStatisticsDirectory / ("pattern_stats_" + std::to_string(patternSize) + ".dat");

    return std::fstream(dumpFileLocation, std::ios::binary | std::ios::in);
}

// Need to open the fstream elsewhere, because it would close the file when going out of scope
SpinImage::utilities::FileDecompressionStream<FileEntry, 8> openDecompressionStream(std::fstream* inputStream) {
    // Sanity check: make sure the file ID bytes are correct
    char fileHeader[4];
    inputStream->read(fileHeader, 4);
    assert(std::string(fileHeader) == "PCF");

    // Buffer sizes
    size_t fileEntryCount;
    inputStream->read(reinterpret_cast<char *>(&fileEntryCount), sizeof(size_t));
    size_t compressedFileEntryListSize;
    inputStream->read(reinterpret_cast<char *>(&compressedFileEntryListSize), sizeof(size_t));

    return SpinImage::utilities::FileDecompressionStream<FileEntry, 8>(inputStream, compressedFileEntryListSize, fileEntryCount);
}

inline bool isParent(const QuiccImage &parent, const QuiccImage &child, const unsigned int limit) {
    unsigned int differentPixelCount = 0;
    for(int i = 0; i < parent.size(); i++) {
        differentPixelCount += std::bitset<32>(parent.at(i) ^ child.at(i)).count();
        if(differentPixelCount > limit) {
            return false;
        }
    }
    return true;
}

void constructIndexTree(std::experimental::filesystem::path quicciImageDumpDirectory,
                        std::experimental::filesystem::path indexDumpDirectory,
                        size_t cachedPatternLimit,
                        size_t fileStartIndex, size_t fileEndIndex) {
    std::fstream histogramFile(indexDumpDirectory / "parents.csv", std::ios::out);
    std::mutex csvFileLock;

    #pragma omp parallel for schedule(dynamic)
    for(int patternSize = 2; patternSize <= 4096; patternSize++) {
        size_t imageBufferLength = cachedPatternLimit / omp_get_num_threads();
        std::set<QuiccImage> imageBuffer;

        std::array<size_t, 4096> parentLevelCountHistogram;
        std::fill(parentLevelCountHistogram.begin(), parentLevelCountHistogram.end(), 0);

        std::fstream patternFileStream = openStatisticsFile(indexDumpDirectory, patternSize);
        SpinImage::utilities::FileDecompressionStream<FileEntry, 8> inStream = openDecompressionStream(&patternFileStream);

        while(!inStream.isDepleted()) {
            std::array<FileEntry, 8> uncompressedBuffer;

            for(size_t block = 0; block < imageBufferLength / 8 && !inStream.isDepleted(); block++) {
                unsigned int bufferImageCount = inStream.read(uncompressedBuffer);
                for(unsigned int i = 0; i < bufferImageCount; i++) {
                    imageBuffer.insert(uncompressedBuffer.at(i).image);
                }
            }

            // Attempt to locate images in previous layer
            // Can only be done if said previous layer exists
            for(size_t parentPatternSize = patternSize - 1; parentPatternSize >= 1 && !imageBuffer.empty(); parentPatternSize--) {
                std::fstream parentPatternFile = openStatisticsFile(indexDumpDirectory, parentPatternSize);
                SpinImage::utilities::FileDecompressionStream<FileEntry, 8> parentStream = openDecompressionStream(&parentPatternFile);
                while(!parentStream.isDepleted()) {
                    std::array<FileEntry, 8> parentBuffer;
                    unsigned int parentCount = parentStream.read(parentBuffer);
                    for(int i = 0; i < parentCount; i++) {
                        std::set<QuiccImage>::iterator it = imageBuffer.begin();

                        while(it != imageBuffer.end()) {
                            if(isParent(*it, parentBuffer.at(i).image, patternSize - parentPatternSize)) {
                                parentLevelCountHistogram.at(parentPatternSize - 1)++;
                                it = imageBuffer.erase(it);
                            } else {
                                it++;
                            }
                        }
                    }
                }
            }
        }
        csvFileLock.lock();
        histogramFile << patternSize << ", ";
        for(int i = 0; i < 4096; i++) {
            histogramFile << parentLevelCountHistogram.at(i) << ", ";
        }
        histogramFile << std::endl;
        csvFileLock.unlock();
        std::cout << patternSize << " complete" << std::endl;
    }
}


