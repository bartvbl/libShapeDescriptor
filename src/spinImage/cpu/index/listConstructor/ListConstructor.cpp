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
#include <condition_variable>
#include <spinImage/cpu/index/IndexIO.h>
#include "ListConstructor.h"

const unsigned int writeBufferSize = 32;

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

    std::mutex loadQueueLock;
    std::condition_variable loadQueueConditionVariable;

    std::array<size_t, spinImageWidthPixels * spinImageWidthPixels> totalPixelTally;
    std::fill(totalPixelTally.begin(), totalPixelTally.end(), 0);

    for(int pixelBatchStartIndex = 0; pixelBatchStartIndex < spinImageWidthPixels * spinImageWidthPixels; pixelBatchStartIndex += openFileLimit) {
        int pixelBatchEndIndex = std::min<int>(pixelBatchStartIndex + openFileLimit, spinImageWidthPixels * spinImageWidthPixels - 1);
        std::vector<std::fstream *> outputStreams;
        std::vector<SpinImage::utilities::FileCompressionStream<IndexEntry, writeBufferSize>> compressionStreams;

        std::experimental::filesystem::path listDirectory = indexDumpDirectory / "lists";
        std::experimental::filesystem::create_directories(listDirectory);

        // Open file streams
        for (int pixelIndex = pixelBatchStartIndex; pixelIndex < pixelBatchEndIndex; pixelIndex++) {
            std::cout << "\rOpening file streams.. " << pixelIndex + 1 << "/" << pixelBatchEndIndex << std::flush;
            std::string fileName = "list_" + std::to_string(pixelIndex) + ".dat";
            std::experimental::filesystem::path outputFile = listDirectory / fileName;
            outputStreams.push_back(new std::fstream(outputFile, std::ios::out | std::ios::binary));
            int outputStreamIndex = pixelIndex - pixelBatchStartIndex;

            // Ensure file opened successfully
            assert(outputStreams.at(outputStreamIndex)->is_open());

            // Write file headers
            const char* headerID = "PXLST";
            outputStreams.at(outputStreamIndex)->write(headerID, 5);
            // Dummy values for compressed buffer size and total reference count
            // Space is allocated for them here, and will be written at the end
            const size_t dummyValue = 0xCDCDCDCDCDCDCDCD;
            outputStreams.at(outputStreamIndex)->write(reinterpret_cast<const char *>(&dummyValue), 8);
            outputStreams.at(outputStreamIndex)->write(reinterpret_cast<const char *>(&dummyValue), 8);

            // Wrapping opened fstream in a compression stream
            compressionStreams.emplace_back(outputStreams.at(outputStreamIndex));
            compressionStreams.at(outputStreamIndex).open();
        }
        std::cout << std::endl;

        unsigned int nextFileIndex = fileStartIndex;

        // Cannot be parallel with a simple OpenMP pragma alone; files MUST be processed in order
        // Also, images MUST be inserted into output files in order
        #pragma omp parallel for schedule(dynamic)
        for (unsigned int fileIndex = fileStartIndex; fileIndex < fileEndIndex; fileIndex++) {

            // Reading image dump file
            std::experimental::filesystem::path path = filesToIndex.at(fileIndex);
            const std::string archivePath = path.string();
            SpinImage::cpu::QUICCIImages images = SpinImage::read::QUICCImagesFromDumpFile(archivePath);

            double totalImageDurationMilliseconds = 0;
            std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();
            int previousDashCount = -1;

            while(nextFileIndex != fileIndex) {
                std::unique_lock<std::mutex> queueLock(loadQueueLock);
                loadQueueConditionVariable.wait_until(queueLock,
                        std::chrono::steady_clock::now() + std::chrono::milliseconds (10));
            }

            #pragma omp parallel
            {
                // Compute thread's job allocation
                unsigned int pixelsPerThread = std::max(
                        (pixelBatchEndIndex - pixelBatchStartIndex) / omp_get_num_threads(), 1);
                unsigned int threadStartPixelIndex = pixelBatchStartIndex + std::min<unsigned int>(
                        omp_get_thread_num() * pixelsPerThread, pixelBatchEndIndex);
                unsigned int threadEndPixelIndex = omp_get_thread_num() + 1 == omp_get_num_threads() ?
                                                   pixelBatchEndIndex
                                                   : pixelBatchStartIndex + std::min<unsigned int>(
                                                           (omp_get_thread_num() + 1) * pixelsPerThread,
                                                           pixelBatchEndIndex);
                unsigned int threadPixelCount = threadEndPixelIndex - threadStartPixelIndex;

                assert(threadEndPixelIndex >= threadStartPixelIndex);
                assert(threadStartPixelIndex >= pixelBatchStartIndex);
                assert(threadEndPixelIndex <= pixelBatchEndIndex);

                // Allocate buffers
                std::vector<std::array<IndexEntry, writeBufferSize>> threadCompressionBuffers;
                std::vector<unsigned int> threadCompressionBufferLengths;

                threadCompressionBuffers.resize(threadPixelCount);
                threadCompressionBufferLengths.resize(threadPixelCount);
                std::fill(threadCompressionBufferLengths.begin(), threadCompressionBufferLengths.end(), 0);

                // For each image, register pixels in dump file
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

                    // Iterate over pixels
                    for(unsigned int pixelIndex = threadStartPixelIndex; pixelIndex < threadEndPixelIndex; pixelIndex++) {
                        unsigned int pixel = SpinImage::index::pattern::pixelAt(combinedImage, pixelIndex);
                        if (pixel == 1) {
                            unsigned int threadBufferIndex = pixelIndex - threadStartPixelIndex;
                            unsigned int currentBufferLength = threadCompressionBufferLengths.at(threadBufferIndex);
                            threadCompressionBuffers.at(threadBufferIndex).at(currentBufferLength) = entry;
                            threadCompressionBufferLengths.at(threadBufferIndex)++;
                            currentBufferLength++;

                            // Only a single thread touches this at any time, thus no guard is needed
                            totalPixelTally.at(pixelIndex)++;

                            // Buffer is full, write it to disk
                            if(currentBufferLength == writeBufferSize) {
                                unsigned int batchBufferIndex = pixelIndex - pixelBatchStartIndex;
                                compressionStreams.at(batchBufferIndex).write(
                                        threadCompressionBuffers.at(threadBufferIndex), writeBufferSize);
                                threadCompressionBufferLengths.at(threadBufferIndex) = 0;
                            }
                        }
                    }

                    std::chrono::steady_clock::time_point imageEndTime = std::chrono::steady_clock::now();
                    #pragma omp atomic
                    totalImageDurationMilliseconds += std::chrono::duration_cast<std::chrono::nanoseconds>(
                            imageEndTime - imageStartTime).count() / 1000000.0;
                }

                // Batch finished. Flush buffers.
                for(unsigned int pixelIndex = threadStartPixelIndex; pixelIndex < threadEndPixelIndex; pixelIndex++) {
                    unsigned int threadBufferIndex = pixelIndex - threadStartPixelIndex;
                    unsigned int batchBufferIndex = pixelIndex - pixelBatchStartIndex;
                    unsigned int bufferEntryCount = threadCompressionBufferLengths.at(threadBufferIndex);

                    if(bufferEntryCount != 0) {
                        compressionStreams.at(batchBufferIndex).write(
                                threadCompressionBuffers.at(threadBufferIndex), bufferEntryCount);
                    }
                }
            }

            std::chrono::steady_clock::time_point endTime = std::chrono::steady_clock::now();
            double durationMilliseconds =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count() / 1000000.0;

            std::cout << "Added file " << (fileIndex + 1) << "/" << fileEndIndex
                      << ": " << archivePath
                      << ", Duration: " << (durationMilliseconds / 1000.0) << "s"
                      << ", Image count: " << images.imageCount << std::endl;

            // Let net thread in. Must be done after the file has been processed
            nextFileIndex++;

            // Wake up other threads
            loadQueueConditionVariable.notify_all();

            // Necessity to prevent libc from hogging all system memory
            malloc_trim(0);

            delete[] images.horizontallyIncreasingImages;
            delete[] images.horizontallyDecreasingImages;
        }

        #pragma omp parallel for schedule(dynamic)
        for (int streamIndex = 0; streamIndex < compressionStreams.size(); streamIndex++) {
            int pixelIndex = pixelBatchStartIndex + streamIndex;
            std::cout << "\rFinalising compression streams.. " +
                    std::to_string(pixelBatchStartIndex + streamIndex + 1) + "/" +
                    std::to_string(spinImageWidthPixels * spinImageWidthPixels) << std::flush;
            compressionStreams.at(streamIndex).close();

            // Write header
            outputStreams.at(streamIndex)->seekp(5);
            outputStreams.at(streamIndex)->write(
                    reinterpret_cast<const char *>(&totalPixelTally.at(pixelIndex)), sizeof(size_t));
            size_t totalCompressedFileSize = compressionStreams.at(streamIndex).getTotalWrittenCompressedBytes();
            outputStreams.at(streamIndex)->write(
                    reinterpret_cast<const char *>(&totalCompressedFileSize), sizeof(size_t));

            outputStreams.at(streamIndex)->close();
            delete outputStreams.at(streamIndex);
            malloc_trim(0);
        }

        std::cout << std::endl;
    }

    // Final construction of the index
    Index index(indexDumpDirectory, &filesToIndex);

    // Write the root node to disk
    std::cout << "Writing core index file.." << std::endl;
    SpinImage::index::io::writeIndex(index, indexDumpDirectory);
}