#include <spinImage/libraryBuildSettings.h>
#include <iostream>
#include <fstream>
#include <cassert>
#include <spinImage/cpu/index/types/WeightedIndexEntry.h>
#include <fast-lzma2.h>
#include <algorithm>
#include <spinImage/cpu/index/types/IndexEntry.h>
#include "ListSorter.h"

void sortListFiles(std::experimental::filesystem::path &indexDumpDirectory) {

    std::experimental::filesystem::path unsortedListRootDirectory = indexDumpDirectory / "lists";
    std::experimental::filesystem::path sortedListRootDirectory = indexDumpDirectory / "sorted_lists";

    #pragma omp parallel
    #pragma omp single
    for (int col = 0; col < spinImageWidthPixels; col++) {
        for (int row = 0; row < spinImageWidthPixels; row++) {
            #pragma omp task
            {
                std::experimental::filesystem::path unsortedListDirectory =
                        unsortedListRootDirectory / std::to_string(col) / std::to_string(row);
                std::experimental::filesystem::path sortedListDirectory =
                        sortedListRootDirectory / std::to_string(col) / std::to_string(row);
                std::experimental::filesystem::create_directories(sortedListDirectory);

                unsigned int possiblePatternLengthCount = spinImageWidthPixels - row;

                for (int patternLength = 0; patternLength < possiblePatternLengthCount; patternLength++) {
                    std::string fileName = "list_" + std::to_string(patternLength) + ".dat";
                    std::experimental::filesystem::path unsortedListFile = unsortedListDirectory / fileName;
                    std::experimental::filesystem::path sortedListFile = sortedListDirectory / fileName;

                    std::fstream inStream(unsortedListFile, std::ios::binary | std::ios::in);

                    char headerID[6] = {0, 0, 0, 0, 0, 0};
                    inStream.read(headerID, 5);
                    assert(std::string(headerID) == "PXLST");

                    size_t indexEntryCount = 0xCDCDCDCDCDCDCDCD;
                    size_t compressedBufferSize = 0xCDCDCDCDCDCDCDCD;

                    inStream.read(reinterpret_cast<char *>(&indexEntryCount), sizeof(size_t));
                    inStream.read(reinterpret_cast<char *>(&compressedBufferSize), sizeof(size_t));

                    std::vector<WeightedIndexEntry> decompressedBuffer;
                    decompressedBuffer.resize(indexEntryCount);
                    char *compressedBuffer = new char[compressedBufferSize];

                    inStream.read(compressedBuffer, compressedBufferSize);
                    inStream.close();

                    FL2_decompress(
                           (void *) decompressedBuffer.data(), indexEntryCount * sizeof(WeightedIndexEntry),
                           (void *) compressedBuffer, compressedBufferSize);

                    delete[] compressedBuffer;

                    std::cout << sortedListFile.string() << " (" << indexEntryCount << " entries)" << std::endl;

                    std::sort(decompressedBuffer.begin(), decompressedBuffer.end());

                    std::vector<IndexEntry> sortedIndexEntryList;
                    sortedIndexEntryList.reserve(decompressedBuffer.size());
                    std::vector<unsigned short> totalImagePixelCounts;
                    totalImagePixelCounts.reserve(spinImageWidthPixels * spinImageWidthPixels);
                    std::vector<unsigned int> pixelCountOccurrenceCounts;
                    pixelCountOccurrenceCounts.reserve(spinImageWidthPixels * spinImageWidthPixels);

                    // Shrinking the buffer by filtering out the total image pixel count
                    // Instead storing the information in a file header
                    int currentTotalPixelCount = 0;
                    unsigned int currentImageTally = 0;
                    for(const auto &entry : decompressedBuffer) {
                        if(entry.remainingPixelCount != currentTotalPixelCount) {
                            if(currentImageTally > 0) {
                                totalImagePixelCounts.push_back(currentTotalPixelCount);
                                pixelCountOccurrenceCounts.push_back(currentImageTally);
                            }
                            currentTotalPixelCount = entry.remainingPixelCount;
                            currentImageTally = 0;
                        }
                        currentImageTally++;
                        sortedIndexEntryList.emplace_back(entry.fileIndex, entry.imageIndex);
                    }
                    if(currentImageTally > 0) {
                        totalImagePixelCounts.push_back(currentTotalPixelCount);
                        pixelCountOccurrenceCounts.push_back(currentImageTally);
                    }

                    size_t outputBufferSize = FL2_compressBound(indexEntryCount * sizeof(IndexEntry));
                    char *compressedOutputBuffer = new char[outputBufferSize];

                    size_t compressedFileSize = FL2_compress(
                            (void *) compressedOutputBuffer, outputBufferSize,
                            (void *) sortedIndexEntryList.data(), indexEntryCount * sizeof(IndexEntry),
                            LZMA2_COMPRESSION_LEVEL);

                    unsigned short totalUniquePixelCount = totalImagePixelCounts.size();
                    size_t decompressedHeaderBufferSize = totalUniquePixelCount * (sizeof(unsigned int) + sizeof(unsigned short));
                    size_t compressedHeaderBufferSize = FL2_compressBound(decompressedHeaderBufferSize);
                    char* headerBuffer = new char[decompressedHeaderBufferSize];
                    char* compressedHeaderBuffer = new char[compressedHeaderBufferSize];

                    char* headerBufferPointer = headerBuffer;
                    for(int i = 0; i < totalUniquePixelCount; i++) {
                        *headerBufferPointer = totalImagePixelCounts.at(i);
                        headerBufferPointer += sizeof(unsigned short);
                        *headerBufferPointer = pixelCountOccurrenceCounts.at(i);
                        headerBufferPointer += sizeof(unsigned int);
                    }
                    assert(headerBufferPointer - headerBuffer == decompressedHeaderBufferSize);

                    size_t compressedHeaderSize = FL2_compress(
                        compressedHeaderBuffer, compressedHeaderBufferSize,
                        headerBuffer, decompressedHeaderBufferSize,
                        LZMA2_COMPRESSION_LEVEL);

                    std::fstream outStream(sortedListFile, std::ios::binary | std::ios::out);
                    outStream.write(headerID, 5);
                    outStream.write(reinterpret_cast<const char *>(&indexEntryCount), sizeof(size_t));
                    outStream.write(reinterpret_cast<const char *>(&totalUniquePixelCount), sizeof(unsigned short));
                    outStream.write(reinterpret_cast<const char *>(&compressedHeaderSize), sizeof(size_t));
                    outStream.write(reinterpret_cast<const char *>(&compressedFileSize), sizeof(size_t));
                    outStream.write(compressedHeaderBuffer, compressedHeaderSize);
                    outStream.write(compressedOutputBuffer, compressedFileSize);
                    outStream.close();

                    delete[] compressedOutputBuffer;
                    delete[] headerBuffer;
                    delete[] compressedHeaderBuffer;
                }
            }
        }
    }

    #pragma omp taskwait


}