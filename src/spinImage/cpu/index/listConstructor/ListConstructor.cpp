#include <spinImage/libraryBuildSettings.h>
#include <iostream>
#include <spinImage/utilities/compression/FileCompressionStream.h>
#include <spinImage/cpu/index/types/IndexEntry.h>
#include <spinImage/utilities/fileutils.h>
#include <malloc.h>
#include <spinImage/cpu/types/QUICCIImages.h>
#include <spinImage/utilities/readers/quicciReader.h>
#include <bitset>
#include <mutex>
#include <spinImage/cpu/index/Pattern.h>
#include <omp.h>
#include <spinImage/cpu/index/IndexIO.h>
#include "ListConstructor.h"

const unsigned int writeBufferSize = 32;

struct OutputBuffer {
    size_t totalWrittenEntryCount = 0;
    unsigned int length = 0;
    std::array<IndexEntry, writeBufferSize> buffer;
    SpinImage::utilities::FileCompressionStream<IndexEntry, writeBufferSize> outStream;
    std::fstream* fileStream;

    explicit OutputBuffer(std::fstream* stream) : outStream(stream), fileStream(stream) {
        assert(fileStream->is_open());

        // Write file headers
        const char* headerID = "PXLST";
        fileStream->write(headerID, 5);
        // Dummy values for compressed buffer size and total reference count
        // Space is allocated for them here, and will be written at the end
        const size_t dummyValue = 0xCDCDCDCDCDCDCDCD;
        fileStream->write(reinterpret_cast<const char *>(&dummyValue), 8);
        fileStream->write(reinterpret_cast<const char *>(&dummyValue), 8);
    }

    void insert(IndexEntry &entry) {
        buffer.at(length) = entry;
        length++;
        totalWrittenEntryCount++;

        if(length == writeBufferSize) {
            outStream.write(buffer, length);
            length = 0;
        }
    }

    void open() {
        outStream.open();
    }

    void close() {
        outStream.write(buffer, length);
        outStream.close();

        fileStream->seekp(5);
        fileStream->write(reinterpret_cast<const char *>(&totalWrittenEntryCount), sizeof(size_t));
        size_t totalCompressedFileSize = outStream.getTotalWrittenCompressedBytes();
        fileStream->write(reinterpret_cast<const char *>(&totalCompressedFileSize), sizeof(size_t));

        fileStream->close();
        delete fileStream;
    }
};

void printProgressBar(const unsigned int imageCount, int &previousDashCount, IndexImageID imageIndex) {
    // Only update the progress bar when needed
    int dashCount = int((float(imageIndex) / float(imageCount)) * 25.0f) + 1;
    if (dashCount > previousDashCount) {
        previousDashCount = dashCount;
        std::stringstream progressBar;
        progressBar << "\r[";

        for (int i = 0; i < 25; i++) {
            progressBar << ((i < dashCount) ? "=" : " ");
        }
        progressBar << "] " << imageIndex << "/" << imageCount << "\r";
        std::cout << progressBar.str() << std::flush;
    }
}

void buildInitialPixelLists(
        const std::experimental::filesystem::path &quicciImageDumpDirectory,
        std::experimental::filesystem::path &indexDumpDirectory,
        size_t openFileLimit,
        size_t fileStartIndex, size_t fileEndIndex) {

    std::cout << "Listing files.." << std::endl;
    std::vector<std::experimental::filesystem::path> filesToIndex = SpinImage::utilities::listDirectory(quicciImageDumpDirectory);
    std::cout << "\tFound " << filesToIndex.size() << " files." << std::endl;

    omp_set_nested(1);

    std::vector<std::vector<std::vector<OutputBuffer>>> outputBuffers;
    std::array<std::array<std::mutex, spinImageWidthPixels>, spinImageWidthPixels> outputBufferLocks;

    std::experimental::filesystem::path listDirectory = indexDumpDirectory / "lists";
    std::experimental::filesystem::create_directories(listDirectory);

    for(int startColumn = 0; startColumn < spinImageWidthPixels; startColumn += openFileLimit) {
        int endColumn = std::min<int>(startColumn + openFileLimit, spinImageWidthPixels * spinImageWidthPixels);
        // Open file streams
        for (int col = 0; col < spinImageWidthPixels; col++) {
            outputBuffers.emplace_back();
            for (int row = 0; row < spinImageWidthPixels; row++) {
                outputBuffers.at(col).emplace_back();

                if(col < startColumn || col >= endColumn) {
                    continue;
                }

                std::experimental::filesystem::path outputDirectory =
                        listDirectory / std::to_string(col) / std::to_string(row);
                std::experimental::filesystem::create_directories(outputDirectory);

                unsigned int possiblePatternLengthCount = spinImageWidthPixels - row;

                for (int patternLength = 0; patternLength < possiblePatternLengthCount; patternLength++) {
                    std::cout << "\rOpening file streams.. column " << col << "/64, row " << row
                              << "/64, pattern length " << patternLength << "/" << possiblePatternLengthCount << "     "
                              << std::flush;

                    std::string fileName = "list_" + std::to_string(patternLength) + ".dat";
                    std::experimental::filesystem::path outputFile = outputDirectory / fileName;
                    outputBuffers.at(col).at(row).emplace_back(
                            new std::fstream(outputFile, std::ios::out | std::ios::binary));
                    outputBuffers.at(col).at(row).at(patternLength).open();
                }
            }
        }
        std::cout << std::endl;

        unsigned int nextFileIndex = fileStartIndex;

        // Cannot be parallel with a simple OpenMP pragma alone; files MUST be processed in order
        // Also, images MUST be inserted into output files in order
        #pragma omp parallel for schedule(dynamic)
        for (unsigned int fileIndex = fileStartIndex; fileIndex < fileEndIndex; fileIndex++) {

            // Reading image dump file
            std::experimental::filesystem::path archivePath = filesToIndex.at(fileIndex);
            SpinImage::cpu::QUICCIImages images = SpinImage::read::QUICCImagesFromDumpFile(archivePath);

            double totalImageDurationMilliseconds = 0;
            std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();
            int previousDashCount = -1;

            #pragma omp critical
            {
                // For each image, register pixels in dump file
                #pragma omp parallel for schedule(dynamic)
                for (IndexImageID imageIndex = 0; imageIndex < images.imageCount; imageIndex++) {
                    printProgressBar(images.imageCount, previousDashCount, imageIndex);
                    std::chrono::steady_clock::time_point imageStartTime = std::chrono::steady_clock::now();

                    QuiccImage combinedImage = combineQuiccImages(
                            images.horizontallyIncreasingImages[imageIndex],
                            images.horizontallyDecreasingImages[imageIndex]);

                    // Count total number of bits in image
                    // Needed during querying to sort search results
                    unsigned short bitCount = 0;
                    for (unsigned int i : combinedImage) {
                        bitCount += std::bitset<32>(i).size();
                    }

                    IndexEntry entry = {fileIndex, imageIndex, bitCount};

                    // Find and process set bit sequences
                    for (int col = startColumn; col < endColumn; col++) {
                        unsigned int patternLength = 0;
                        unsigned int patternStartRow = 0;
                        bool previousPixelWasSet = false;

                        for (int row = 0; row < spinImageWidthPixels; row++) {
                            int pixel = SpinImage::index::pattern::pixelAt(combinedImage, row, col);
                            if (pixel == 1) {
                                if (previousPixelWasSet) {
                                    // Pattern turned out to be one pixel longer
                                    patternLength++;
                                } else {
                                    // We found a new pattern
                                    patternStartRow = row;
                                    patternLength = 1;
                                }
                            } else if (previousPixelWasSet) {
                                // Previous pixel was set, but this one is not
                                // This is thus a pattern that ended here.
                                outputBufferLocks.at(col).at(patternStartRow).lock();
                                outputBuffers.at(col).at(patternStartRow).at(patternLength - 1).insert(entry);
                                outputBufferLocks.at(col).at(patternStartRow).unlock();
                                patternLength = 0;
                                patternStartRow = 0;
                            }

                            previousPixelWasSet = pixel == 1;
                        }

                        if (previousPixelWasSet) {
                            outputBufferLocks.at(col).at(patternStartRow).lock();
                            outputBuffers.at(col).at(patternStartRow).at(patternLength - 1).insert(entry);
                            outputBufferLocks.at(col).at(patternStartRow).unlock();
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

            std::cout << "Added file " << (fileIndex + 1) << "/" << fileEndIndex
                      << ": " << archivePath
                      << ", Duration: " << (durationMilliseconds / 1000.0) << "s"
                      << ", Image count: " << images.imageCount << std::endl;

            // Necessity to prevent libc from hogging all system memory
            malloc_trim(0);

            delete[] images.horizontallyIncreasingImages;
            delete[] images.horizontallyDecreasingImages;
        }

        #pragma omp parallel for schedule(dynamic) collapse(2)
        for (int col = startColumn; col < endColumn; col++) {
            for (int row = 0; row < spinImageWidthPixels; row++) {
                unsigned int possiblePatternLengthCount = spinImageWidthPixels - row;
                for (int patternLength = 0; patternLength < possiblePatternLengthCount; patternLength++) {
                    std::cout << "\rClosing file streams.. "
                                 "column " + std::to_string(col) + "/64, "
                                                                   "row " + std::to_string(row) + "/64, "
                                                                                                  "pattern length " +
                                 std::to_string(patternLength) + "/" +
                                 std::to_string(possiblePatternLengthCount) + "     " << std::flush;

                    // Flush output buffers and close file stream
                    outputBuffers.at(col).at(row).at(patternLength).close();
                }

                // Trim memory
                malloc_trim(0);
            }
        }
        std::cout << std::endl;
    }

    // Final construction of the index
    Index index(indexDumpDirectory, &filesToIndex);

    // Write the root node to disk
    std::cout << "Writing core index file.." << std::endl;
    SpinImage::index::io::writeIndex(index, indexDumpDirectory);
}