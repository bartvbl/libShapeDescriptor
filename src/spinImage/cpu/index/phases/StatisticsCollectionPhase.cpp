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
#include <unordered_map>

// First phase. Produces a file, one for each pattern length, with the following:
// - A list of all unique patterns
// - How often each pattern occurred in the dataset
// - The number of outgoing shortcut links from that pattern

struct PatternEntry {
    QuiccImage patternImage;
    size_t occurrenceCount = 0;
};

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

    size_t totalImageCount = 0;
    size_t currentTotalImageIndex = 0;

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
                totalImageCount += images.imageCount;
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
                        while(!SpinImage::index::pattern::findNext(
                                combined, patternImage, patternSize, row, col, floodFillBuffer)) {
                            if(patternSize - 1 >= minSize
                            && patternSize - 1 < maxSize) {
                                std::map<QuiccImage, size_t>::iterator item = seenPatterns.at(patternSize - 1).find(patternImage);
                                if(item == seenPatterns.at(patternSize - 1).end()) {
                                    // Make a note of the image, start usage count at 1
                                    threadSeenPatterns.at(patternSize - 1).insert({patternImage, 1});
                                } else {
                                    // Increment usage count
                                    item->second++;
                                }
                            }

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

                if(fileIndex % 10 == 9) {
                    std::cout << "\rUnique pattern counts: " << std::endl;
                    for(int i = 0; i < minSize; i++) {
                        std::cout << countedPatterns.at(i) << ",";
                    }
                    for(int i = minSize; i < maxSize; i++) {
                        countedPatterns.at(i) = seenPatterns.at(i).size();
                        std::cout << seenPatterns.at(i).size() << ",";
                    }
                    std::cout << std::endl;
                    std::cout << "Total pattern counts: " << std::endl;
                    for(int i = 0; i < maxSize; i++) {
                        std::cout << totalPatternOccurrenceCounts.at(i) << ",";
                    }
                    std::cout << std::endl;
                }

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

        std::cout << "Unique pattern counts: " << std::endl;
        for(int i = 0; i < minSize; i++) {
            std::cout << countedPatterns.at(i) << ",";
        }
        for(int i = minSize; i < maxSize; i++) {
            countedPatterns.at(i) = seenPatterns.at(i).size();
            std::cout << seenPatterns.at(i).size() << ",";
        }
        std::cout << std::endl;
        std::cout << "Total pattern counts: " << std::endl;
        for(int i = 0; i < maxSize; i++) {
            std::cout << totalPatternOccurrenceCounts.at(i) << ",";
        }
        std::cout << std::endl;

        malloc_trim(0);

    }

    std::cout << std::endl << "Total Added Image Count: " << totalImageCount << std::endl;
}