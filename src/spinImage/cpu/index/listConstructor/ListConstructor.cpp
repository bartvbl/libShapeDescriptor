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
#include "ListConstructor.h"

const unsigned int writeBufferSize = 32;

void buildSimpleListIndex(
        const std::experimental::filesystem::path &quicciImageDumpDirectory,
        std::experimental::filesystem::path &indexDumpDirectory,
        size_t openFileLimit,
        size_t fileStartIndex, size_t fileEndIndex) {

    std::cout << "Listing files.." << std::endl;
    std::vector<std::experimental::filesystem::path> filesToIndex = SpinImage::utilities::listDirectory(quicciImageDumpDirectory);
    std::cout << "\tFound " << filesToIndex.size() << " files." << std::endl;

    for(int startIndex = 0; startIndex < spinImageWidthPixels * spinImageWidthPixels; startIndex += openFileLimit) {
        int endIndex = std::min<int>(startIndex + openFileLimit, spinImageWidthPixels * spinImageWidthPixels);
        std::vector<std::fstream *> outputStreams;
        std::vector<SpinImage::utilities::FileCompressionStream<IndexEntry, writeBufferSize>> compressionStreams;
        outputStreams.reserve(spinImageWidthPixels * spinImageWidthPixels);
        compressionStreams.reserve(spinImageWidthPixels * spinImageWidthPixels);

        std::experimental::filesystem::path listDirectory = indexDumpDirectory / "lists";
        std::experimental::filesystem::create_directories(listDirectory);

        // Open file streams
        std::cout << "Opening file streams.." << std::endl;
        for (int i = startIndex; i < endIndex; i++) {
            std::cout << i << std::endl;
            std::string fileName = "list_" + std::to_string(i) + ".dat";
            std::experimental::filesystem::path outputFile = listDirectory / fileName;
            outputStreams.push_back(new std::fstream(outputFile, std::ios::out | std::ios::binary));
            // Ensure file opened successfully
            assert(outputStreams.at(i - startIndex)->is_open());
        }

        std::cout << "Initialising compression streams.." << std::endl;
        for (int i = startIndex; i < endIndex; i++) {
            std::cout << i << std::endl;
            compressionStreams.emplace_back(outputStreams.at(i - startIndex));
        }

        // Cannot be parallel with a simple OpenMP pragma; files MUST be processed in order
        for (unsigned int fileIndex = fileStartIndex; fileIndex < fileEndIndex; fileIndex++) {
            std::experimental::filesystem::path path = filesToIndex.at(fileIndex);
            const std::string archivePath = path.string();
            std::cout << "\tReading file: " << archivePath << std::endl;
            SpinImage::cpu::QUICCIImages images = SpinImage::read::QUICCImagesFromDumpFile(archivePath);
            double totalImageDurationMilliseconds = 0;
            {
                std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();
                int previousDashCount = -1;
                #pragma omp parallel
                {
                    //std::unique_ptr<std::array<std::array<IndexEntry, writeBufferSize>, spinImageWidthPixels * spinImageWidthPixels>> buffers(new std::array<std::array<IndexEntry, writeBufferSize>, spinImageWidthPixels * spinImageWidthPixels>);
                    //std::array<unsigned int, spinImageWidthPixels * spinImageWidthPixels> bufferIndices;
                    unsigned int pixelsPerThread = (spinImageWidthPixels * spinImageWidthPixels) / omp_get_num_threads();
                    unsigned int startPixelIndex = omp_get_thread_num() * pixelsPerThread;
                    unsigned int endPixelIndex = omp_get_thread_num() < omp_get_num_threads() - 1 ?
                                                 (omp_get_thread_num() + 1) * pixelsPerThread
                                                 : spinImageWidthPixels * spinImageWidthPixels;
                    for (IndexImageID imageIndex = 0; imageIndex < images.imageCount; imageIndex++) {
                        // Only update the progress bar when needed
                        int dashCount = int((float(imageIndex) / float(images.imageCount)) * 25.0f) + 1;
                        if (dashCount > previousDashCount) {
                            previousDashCount = dashCount;
                            std::stringstream progressBar;
                            progressBar << "\r[";

                            for (int i = 0; i < 25; i++) {
                                progressBar << ((i < dashCount) ? "=" : " ");
                            }
                            progressBar << "] " << imageIndex << "/" << images.imageCount << "\r";
                            std::cout << progressBar.str() << std::flush;
                        }
                        std::chrono::steady_clock::time_point imageStartTime = std::chrono::steady_clock::now();
                        QuiccImage combinedImage = combineQuiccImages(
                                images.horizontallyIncreasingImages[imageIndex],
                                images.horizontallyDecreasingImages[imageIndex]);

                        unsigned short bitCount = 0;
                        for (unsigned int i : combinedImage) {
                            bitCount += std::bitset<32>(i).size();
                        }

                        IndexEntry entry = {fileIndex, imageIndex, bitCount};

                        std::array<IndexEntry, writeBufferSize> writeBuffer;

                        for(unsigned int pixelIndex = startPixelIndex; pixelIndex < endPixelIndex; pixelIndex++) {
                            unsigned int pixel = SpinImage::index::pattern::pixelAt(combinedImage, pixelIndex);
                            if (pixel == 1 && startIndex <= pixelIndex && pixelIndex < endIndex) {
                                writeBuffer.at(0) = entry;
                                compressionStreams.at(pixelIndex - startIndex).write(writeBuffer, 1);
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
            };

            // Necessity to prevent libc from hogging all system memory
            malloc_trim(0);

            delete[] images.horizontallyIncreasingImages;
            delete[] images.horizontallyDecreasingImages;
        }

        std::cout << "Finalising compression streams.." << std::endl;
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < compressionStreams.size(); i++) {
            std::cout << "\r\t" + std::to_string(startIndex + i + 1) + "/" +
                         std::to_string(spinImageWidthPixels * spinImageWidthPixels) << std::flush;
            compressionStreams.at(i).close();
            malloc_trim(0);
            outputStreams.at(i)->close();
            delete outputStreams.at(i);
        }
    }

    // Final construction of the index
    //Index index(indexDumpDirectory, &filesToIndex);

    // Write the root node to disk
    std::cout << "Writing core index file.." << std::endl;
    //SpinImage::index::io::writeIndex(index, indexDumpDirectory);
}