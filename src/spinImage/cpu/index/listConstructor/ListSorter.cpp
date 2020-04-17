#include <spinImage/libraryBuildSettings.h>
#include <iostream>
#include <fstream>
#include <cassert>
#include <spinImage/cpu/index/types/IndexEntry.h>
#include <fast-lzma2.h>
#include <algorithm>
#include "ListSorter.h"



bool operator== (const IndexEntry& lhs, const IndexEntry& rhs) {
    return
            lhs.remainingPixelCount == rhs.remainingPixelCount &&
                    lhs.fileIndex == rhs.fileIndex &&
                    lhs.imageIndex == rhs.imageIndex;
}

void sortListFiles(std::experimental::filesystem::path &indexDumpDirectory) {

    std::experimental::filesystem::path unsortedListRootDirectory = indexDumpDirectory / "lists";
    std::experimental::filesystem::path sortedListRootDirectory = indexDumpDirectory / "sorted_lists";

    FL2_DCtx* decompressionContext = FL2_createDCtxMt(12);
    FL2_CCtx* compressionContext = FL2_createCCtxMt(12);

    for (int col = 0; col < spinImageWidthPixels; col++) {
        for (int row = 0; row < spinImageWidthPixels; row++) {
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

                size_t indexEntryCount;
                size_t compressedBufferSize;

                inStream.read(reinterpret_cast<char *>(&indexEntryCount), sizeof(size_t));
                inStream.read(reinterpret_cast<char *>(&compressedBufferSize), sizeof(size_t));

                std::vector<IndexEntry> decompressedBuffer;
                decompressedBuffer.resize(indexEntryCount);
                char* compressedBuffer = new char[compressedBufferSize];

                inStream.read(compressedBuffer, compressedBufferSize);
                inStream.close();

                FL2_decompressDCtx(decompressionContext,
                        (void*) decompressedBuffer.data(), indexEntryCount * sizeof(IndexEntry),
                        (void*) compressedBuffer, compressedBufferSize);

                delete[] compressedBuffer;

                std::cout << sortedListFile.string() << " (" << indexEntryCount << " entries)" << std::endl;

                std::sort(decompressedBuffer.begin(), decompressedBuffer.end(), indexEntryComparator);

                std::fstream outStream(sortedListFile, std::ios::binary | std::ios::out);
                outStream.write(headerID, 5);
                outStream.write(reinterpret_cast<const char *>(&indexEntryCount), sizeof(size_t));
                // reserve space for compressed buffer size

                size_t outputBufferSize = FL2_compressBound(indexEntryCount * sizeof(IndexEntry));
                char* compressedOutputBuffer = new char[outputBufferSize];

                size_t compressedFileSize = FL2_compressCCtx(compressionContext,
                        (void*) decompressedBuffer.data(), indexEntryCount * sizeof(IndexEntry),
                        (void*) compressedOutputBuffer, outputBufferSize, LZMA2_COMPRESSION_LEVEL);

                outStream.write(reinterpret_cast<const char *>(&compressedFileSize), sizeof(size_t));
                outStream.write(compressedOutputBuffer, compressedFileSize);

                delete[] compressedOutputBuffer;
                outStream.close();
            }
        }
    }
    FL2_freeCCtx(compressionContext);
    FL2_freeDCtx(decompressionContext);
}