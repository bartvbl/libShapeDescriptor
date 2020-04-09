#include "StatisticsCollectionPhase.h"
#include <spinImage/utilities/fileutils.h>
#include <omp.h>
#include <spinImage/cpu/types/QuiccImage.h>
#include <mutex>
#include <set>
#include <spinImage/utilities/readers/quicciReader.h>
#include <iostream>
#include <spinImage/cpu/index/types/IndexEntry.h>
#include <spinImage/cpu/index/Pattern.h>
#include <malloc.h>
#include <fstream>
#include <fast-lzma2.h>
#include <spinImage/cpu/index/phases/types/FileEntry.h>
#include <cassert>

// First phase. Produces a file, one for each pattern length, with the following:
// - A list of all unique patterns
// - How often each pattern occurred in the dataset
// - The number of outgoing shortcut links from that pattern



void computePatternStatisticsFile(
        std::experimental::filesystem::path quicciImageDumpDirectory,
        std::experimental::filesystem::path indexDumpDirectory,
        size_t cachedPatternLimit,
        size_t fileStartIndex, size_t fileEndIndex) {
    std::vector<std::experimental::filesystem::path> filesInDirectory = SpinImage::utilities::listDirectory(quicciImageDumpDirectory);

    std::experimental::filesystem::path patternStatisticsDirectory(indexDumpDirectory / "patternStats");
    omp_set_nested(1);

    std::vector<std::experimental::filesystem::path>* indexedFiles =
            new std::vector<std::experimental::filesystem::path>();

    /*bool enableStatisticsDump = statisticsFileDumpLocation != std::experimental::filesystem::path("/none/selected");
    if(enableStatisticsDump) {
        std::cout << "Statistics will be dumped to " << statisticsFileDumpLocation << std::endl;
    }*/

    size_t endIndex = fileEndIndex == fileStartIndex ? filesInDirectory.size() : fileEndIndex;

    std::array<size_t, 4096> countedPatterns;
    std::fill(countedPatterns.begin(), countedPatterns.end(), 0);
    std::array<size_t, 4096> totalPatternOccurrenceCounts;
    std::fill(totalPatternOccurrenceCounts.begin(), totalPatternOccurrenceCounts.end(), 0);

    int minSize = 0;
    int maxSize = 4096;
    for(; minSize < 4096; minSize = maxSize) {
        maxSize = 4096;

        std::array<std::mutex, 4096> seenPatternLocks;
        std::array<std::map<QuiccImage, size_t>, 4096> seenPatterns;

        #pragma omp parallel for schedule(dynamic)
        for (unsigned int fileIndex = 0; fileIndex < endIndex; fileIndex++) {
            std::experimental::filesystem::path path = filesInDirectory.at(fileIndex);
            const std::string archivePath = path.string();

            SpinImage::cpu::QUICCIImages images = SpinImage::read::QUICCImagesFromDumpFile(archivePath);

            #pragma omp critical
            {
                indexedFiles->emplace_back(archivePath);
                std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();

                #pragma omp parallel
                {
                    std::cout << "\rRunning.." << std::flush;
                    std::vector<std::pair<unsigned short, unsigned short>> floodFillBuffer;
                    floodFillBuffer.reserve(4096);
                    QuiccImage patternImage;

                    std::array<size_t, 4096> threadTotalSeenPatterns;
                    std::fill(threadTotalSeenPatterns.begin(), threadTotalSeenPatterns.end(), 0);

                    std::array<std::map<QuiccImage, size_t>, 4096> threadSeenPatterns;

                    #pragma omp for schedule(dynamic)
                    for (IndexImageID imageIndex = 0; imageIndex < images.imageCount; imageIndex++) {
                        std::chrono::steady_clock::time_point imageStartTime = std::chrono::steady_clock::now();
                        QuiccImage combined = combineQuiccImages(
                                images.horizontallyIncreasingImages[imageIndex],
                                images.horizontallyDecreasingImages[imageIndex]);

                        unsigned int row = 0;
                        unsigned int col = 0;
                        unsigned int patternSize = 0;
                        while(SpinImage::index::pattern::findNext(
                                combined, patternImage, patternSize, row, col, floodFillBuffer)) {
                            if(patternSize - 1 >= minSize && patternSize - 1 < maxSize) {
                                std::map<QuiccImage, size_t>::iterator item =
                                        seenPatterns.at(patternSize - 1).find(patternImage);
                                if(item == seenPatterns.at(patternSize - 1).end()) {
                                    // Make a note of the image, start usage count at 1
                                    threadSeenPatterns.at(patternSize - 1)[patternImage] = 1;
                                } else {
                                    // Increment usage count
                                    item->second++;
                                }
                            }

                            assert(patternSize > 0);
                            threadTotalSeenPatterns.at(patternSize - 1)++;
                        }

                        std::chrono::steady_clock::time_point imageEndTime = std::chrono::steady_clock::now();
                    }

                    #pragma omp barrier

                    std::cout << "\rCollating results.." << std::flush;
                    for (int i = minSize; i < maxSize; i++) {
                        if (!threadSeenPatterns.at(i).empty()) {
                            seenPatternLocks.at(i).lock();
                            for(const auto& threadEntry : threadSeenPatterns.at(i)) {
                                std::map<QuiccImage, size_t>::iterator seenEntry = seenPatterns.at(i).find(threadEntry.first);
                                if(seenEntry == seenPatterns.at(i).end()) {
                                    seenPatterns.at(i).insert({threadEntry.first, threadEntry.second});
                                } else {
                                    // Merge the two counts
                                    seenEntry->second += threadEntry.second;
                                }
                            }
                            seenPatternLocks.at(i).unlock();
                        }
                        if(threadTotalSeenPatterns.at(i) != 0) {
                            #pragma omp atomic
                            totalPatternOccurrenceCounts.at(i) += threadTotalSeenPatterns.at(i);
                        }
                    }
                }

                std::cout << "\rTrimming cache..        " << std::flush;

                size_t totalPatternCount = 0;
                for(int i = 0; i < 4096; i++) {
                    totalPatternCount += seenPatterns.at(i).size();
                }

                int patternIndex = maxSize - 1;
                while(totalPatternCount > cachedPatternLimit && patternIndex >= 0) {
                    size_t bucketSize = seenPatterns.at(patternIndex).size();
                    totalPatternCount -= bucketSize;
                    if(bucketSize != 0) {
                        std::cout << "\rCache is getting too large. Postponing counting patterns of length " + std::to_string(patternIndex) + " to a later iteration. New pattern count: " + std::to_string(totalPatternCount) + "\n";
                    }
                    // Delete set contents to free up memory
                    seenPatterns.at(patternIndex).clear();
                    // Reset count
                    totalPatternOccurrenceCounts.at(patternIndex) = 0;
                    patternIndex--;
                }
                maxSize = patternIndex + 1;

                std::chrono::steady_clock::time_point endTime = std::chrono::steady_clock::now();
                double durationMilliseconds =
                        std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count() / 1000000.0;

                std::cout << "\rAdded file " << (fileIndex + 1) << "/" << endIndex << " (" << minSize << "-" << maxSize << ")"
                          << ": " << archivePath
                          << ", pattern image count: " << totalPatternCount
                          << ", Duration: " << (durationMilliseconds / 1000.0) << "s"
                          << ", Image count: " << images.imageCount << std::endl;
            };

            // Necessity to prevent libc from hogging all system memory
            if(fileIndex % 50 == 49) {
                malloc_trim(0);
            }

            delete[] images.horizontallyIncreasingImages;
            delete[] images.horizontallyDecreasingImages;
        }

        std::cout << "Iteration complete. Dumping discovered patterns.." << std::endl;
        #pragma omp parallel for schedule(dynamic)
        for(int i = minSize; i < maxSize; i++) {
            FL2_CStream* compressionStream = FL2_createCStream();;
            FL2_initCStream(compressionStream, 9);

            std::experimental::filesystem::path dumpFileLocation =
                    patternStatisticsDirectory / ("pattern_stats_" + std::to_string(i+1) + ".dat");
            std::experimental::filesystem::create_directories(dumpFileLocation.parent_path());
            std::fstream dumpFileStream(dumpFileLocation, std::ios::binary | std::ios::out);

            size_t totalCompressedSize = 0;

            const unsigned int imagesPerBuffer = 8;
            const size_t uncompressedBufferSize = imagesPerBuffer * sizeof(FileEntry);
            const size_t compressedBufferSize = imagesPerBuffer * sizeof(FileEntry);
            unsigned char uBuffer[uncompressedBufferSize];
            unsigned char cBuffer[compressedBufferSize];
            FL2_inBuffer uncompressedBuffer = {uBuffer, uncompressedBufferSize, uncompressedBufferSize};
            FL2_outBuffer compressedBuffer = {cBuffer, compressedBufferSize, 0};

            // Write file header
            const char fileID[4] = "PCF";
            dumpFileStream.write(fileID, 4);
            size_t patternCount = seenPatterns.at(i).size();
            dumpFileStream.write(reinterpret_cast<const char *>(&patternCount), sizeof(size_t));
            // Allocates space for the total compressed size
            dumpFileStream.write(reinterpret_cast<const char *>(&patternCount), sizeof(size_t));
            // Write file contents
            std::map<QuiccImage, size_t>::iterator it = seenPatterns.at(i).begin();
            do {
                if (uncompressedBuffer.pos == uncompressedBuffer.size) {
                    uncompressedBuffer.size = 0;
                    for(int entry = 0; entry < imagesPerBuffer; entry++) {
                        if(it == seenPatterns.at(i).end()) {
                            break;
                        }
                        reinterpret_cast<FileEntry*>(uBuffer)[entry] = {it->first, it->second};
                        uncompressedBuffer.size += sizeof(FileEntry);
                        it++;
                    }
                    uncompressedBuffer.pos = 0;
                }
                FL2_compressStream(compressionStream, &compressedBuffer, &uncompressedBuffer);

                dumpFileStream.write((char*) compressedBuffer.dst, compressedBuffer.pos);
                totalCompressedSize += compressedBuffer.pos;
                compressedBuffer.pos = 0;
            } while (uncompressedBuffer.size == uncompressedBufferSize);
            // Write dictionary / metadata
            unsigned int status;
            do {
                status = FL2_endStream(compressionStream, &compressedBuffer);
                dumpFileStream.write((char*) compressedBuffer.dst, compressedBuffer.pos);
                totalCompressedSize += compressedBuffer.pos;
                compressedBuffer.pos = 0;
            } while (status);
            FL2_freeCStream(compressionStream);
            dumpFileStream.seekp(sizeof(fileID) + sizeof(size_t));
            dumpFileStream.write(reinterpret_cast<const char *>(&totalCompressedSize), sizeof(size_t));
        }
        malloc_trim(0);
    }
}