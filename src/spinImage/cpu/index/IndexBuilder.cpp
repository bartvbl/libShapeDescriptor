#include <spinImage/utilities/fileutils.h>
#include <iostream>
#include <spinImage/cpu/types/QUICCIImages.h>
#include <spinImage/utilities/readers/quicciReader.h>
#include <bitset>
#include <spinImage/cpu/types/QuiccImage.h>
#include <json.hpp>
#include <fstream>
#include "IndexBuilder.h"
#include "tsl/ordered_map.h"
#include "IndexIO.h"

#include <fast-lzma2.h>
#include <malloc.h>
#include <omp.h>
#include <set>

template<class Key, class T, class Ignore, class Allocator,
        class Hash = std::hash<Key>, class KeyEqual = std::equal_to<Key>,
        class AllocatorPair = typename std::allocator_traits<Allocator>::template rebind_alloc<std::pair<Key, T>>,
        class ValueTypeContainer = std::vector<std::pair<Key, T>, AllocatorPair>>
using ordered_map = tsl::ordered_map<Key, T, Hash, KeyEqual, AllocatorPair, ValueTypeContainer>;

using json = nlohmann::basic_json<ordered_map>;

Index SpinImage::index::build(
        std::experimental::filesystem::path quicciImageDumpDirectory,
        std::experimental::filesystem::path indexDumpDirectory,
        size_t cacheNodeLimit,
        size_t cacheImageLimit,
        size_t fileStartIndex,
        size_t fileEndIndex,
        bool appendToExistingIndex,
        std::experimental::filesystem::path statisticsFileDumpLocation) {
    std::vector<std::experimental::filesystem::path> filesInDirectory = SpinImage::utilities::listDirectory(quicciImageDumpDirectory);
    std::cout << "Sizes: " << sizeof(NodeBlock) << ", " << sizeof(NodeBlockEntry) << std::endl;
    std::experimental::filesystem::path indexDirectory(indexDumpDirectory);
    omp_set_nested(1);

    std::vector<std::experimental::filesystem::path>* indexedFiles =
            new std::vector<std::experimental::filesystem::path>();

    if(appendToExistingIndex) {
        Index loadedIndex = SpinImage::index::io::readIndex(indexDumpDirectory);
        indexedFiles->resize(loadedIndex.indexedFileList->size());
        std::copy(loadedIndex.indexedFileList->begin(), loadedIndex.indexedFileList->end(), indexedFiles->begin());
    } else {
        indexedFiles->reserve(filesInDirectory.size());
    }

    bool enableStatisticsDump = statisticsFileDumpLocation != std::experimental::filesystem::path("/none/selected");
    if(enableStatisticsDump) {
        std::cout << "Statistics will be dumped to " << statisticsFileDumpLocation << std::endl;
    }

    size_t endIndex = fileEndIndex == fileStartIndex ? filesInDirectory.size() : fileEndIndex;

    size_t totalImageCount = 0;
    size_t currentTotalImageIndex = 0;

    std::array<size_t, 4096> patternCountZeroes;
    std::fill(patternCountZeroes.begin(), patternCountZeroes.end(), 0);

    QuiccImage zeroImage;
    std::fill(zeroImage.begin(), zeroImage.end(), 0);

    std::array<size_t, 4096> countedPatterns = patternCountZeroes;
    std::array<size_t, 4096> totalPatternOccurrenceCounts = patternCountZeroes;

    std::array<unsigned short, 64> rowOfZeroes;
    std::fill(rowOfZeroes.begin(), rowOfZeroes.end(), 0);


    int minSize = 0;
    int maxSize = 4096;
    for(; minSize < 4096; minSize = maxSize) {
	maxSize = 4096;

        std::array<std::mutex, 4096> seenPatternLocks;
        std::array<std::set<QuiccImage>, 4096> seenPatterns;

        #pragma omp parallel for schedule(dynamic)
        for (unsigned int fileIndex = 0; fileIndex < endIndex; fileIndex++) {
            std::experimental::filesystem::path path = filesInDirectory.at(fileIndex);
            const std::string archivePath = path.string();

            SpinImage::cpu::QUICCIImages images = SpinImage::read::QUICCImagesFromDumpFile(archivePath);
            double totalImageDurationMilliseconds = 0;
            #pragma omp critical
            {
                totalImageCount += images.imageCount;
                indexedFiles->emplace_back(archivePath);
                std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();

                #pragma omp parallel
                {
                    std::cout << "\rRunning.." << std::flush;
                    std::vector<std::pair<unsigned short, unsigned short>> floodFillPixels;
                    floodFillPixels.reserve(4096);
                    QuiccImage patternImage = zeroImage;
                    QuiccImage floodFillImage = zeroImage;
		    std::array<size_t, 4096> threadTotalSeenPatterns = patternCountZeroes;

                    std::array<std::set<QuiccImage>, 4096> threadSeenPatterns;

                    #pragma omp for schedule(dynamic)
                    for (IndexImageID imageIndex = 0; imageIndex < images.imageCount; imageIndex++) {
                        /*if (imageIndex % 5000 == 0) {
                            std::stringstream progressBar;
                            progressBar << "\r[";
                            int dashCount = int((float(imageIndex) / float(images.imageCount)) * 25.0f) + 1;
                            for (int i = 0; i < 25; i++) {
                                progressBar << ((i < dashCount) ? "=" : " ");
                            }
                            progressBar << "] " << imageIndex << "/" << images.imageCount << "\r";
                            std::cout << progressBar.str() << std::flush;
                        }*/

                        std::chrono::steady_clock::time_point imageStartTime = std::chrono::steady_clock::now();
                        QuiccImage combined = combineQuiccImages(
                                images.horizontallyIncreasingImages[imageIndex],
                                images.horizontallyDecreasingImages[imageIndex]);
                        IndexEntry entry = {fileIndex, imageIndex};


                        for (unsigned int row = 0; row < 64; row++) {
                            for (unsigned int col = 0; col < 64; col++) {

                                unsigned int pixel = (unsigned int) ((
                                                         combined.at(2 * row + (col / 32))
                                                                 >> (31U - col)) & 0x1U);

                                if (pixel == 1) {
                                    unsigned int regionSize = 0;
                                    floodFillPixels.clear();
                                    floodFillPixels.emplace_back(row, col);
                                    std::fill(patternImage.begin(), patternImage.end(), 0);
                                    std::fill(floodFillImage.begin(), floodFillImage.end(), 0);

                                    while (!floodFillPixels.empty()) {
                                        std::pair<unsigned short, unsigned short> pixelIndex = floodFillPixels.at(
                                                floodFillPixels.size() - 1);
                                        floodFillPixels.erase(floodFillPixels.begin() + floodFillPixels.size() - 1);
                                        unsigned int chunkIndex = 2 * pixelIndex.first + (pixelIndex.second / 32);
                                        unsigned int chunk = combined.at(chunkIndex);
                                        unsigned int floodPixel = (unsigned int)
                                                ((chunk >> (31U - pixelIndex.second % 32)) & 0x1U);



                                        if (floodPixel == 1) {
                                            regionSize++;
                                            // Add pixel to pattern image
                                            unsigned int bitEnablingMask = 0x1U << (31U - pixelIndex.second % 32);
                                            patternImage.at(chunkIndex) |= bitEnablingMask;
                                            // Disable pixel
                                            unsigned int bitDisablingMask = ~bitEnablingMask;
                                            combined.at(chunkIndex) = chunk & bitDisablingMask;
                                            // Queue surrounding pixels
                                            const int range = 3;
                                            for (int floodRow = std::max(int(pixelIndex.first) - range, 0);
                                                     floodRow <= std::min(63, pixelIndex.first + range);
                                                     floodRow++) {
                                                for (int floodCol = std::max(int(pixelIndex.second) - range, 0);
                                                         floodCol <= std::min(63, pixelIndex.second + range);
                                                         floodCol++) {
                                                    unsigned int childChunkIndex = 2 * floodRow + (floodCol / 32);
                                                    unsigned int childChunk = floodFillImage.at(childChunkIndex);
                                                    unsigned int pixelWasAlreadyVisited = (unsigned int)
                                                            ((childChunk >> (31U - floodCol % 32)) & 0x1U);
                                                    if(pixelWasAlreadyVisited == 0) {
                                                        // Mark the pixel as visited
                                                        unsigned int childMarkMask = 0x1U << (31U - floodCol % 32);
                                                        floodFillImage.at(chunkIndex) |= childMarkMask;
                                                        floodFillPixels.emplace_back(floodRow, floodCol);
                                                    }
                                                }
                                            }
                                        }
                                    }

                                    if(regionSize-1 >= minSize && regionSize-1 < maxSize && seenPatterns.at(regionSize-1).find(patternImage) == seenPatterns.at(regionSize-1).end()) {
                                        threadSeenPatterns.at(regionSize - 1).insert(patternImage);
                                    }

                                    threadTotalSeenPatterns.at(regionSize - 1)++;
                                }
                            }
                        }


                        //cache.insertImage(combined, entry);
                        std::chrono::steady_clock::time_point imageEndTime = std::chrono::steady_clock::now();
                        #pragma omp atomic
                        totalImageDurationMilliseconds += std::chrono::duration_cast<std::chrono::nanoseconds>(
                                imageEndTime - imageStartTime).count() / 1000000.0;
                    }
                    std::cout << "\rAwaiting barrier.." << std::flush;
		    #pragma omp barrier
                    std::cout << "\rCollating results.." << std::flush;
                    for (int i = minSize; i < maxSize; i++) {
                        if (!threadSeenPatterns.at(i).empty()) {
                            seenPatternLocks.at(i).lock();
                            seenPatterns.at(i).insert(threadSeenPatterns.at(i).begin(), threadSeenPatterns.at(i).end());
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
                while(totalPatternCount > cacheImageLimit && patternIndex >= 0) {
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

    // Final construction of the index
    Index index(indexDirectory, indexedFiles);

    // Write the root node to disk
    std::cout << "Writing core index files.." << std::endl;
    SpinImage::index::io::writeIndex(index, indexDirectory);

    return index;
}


