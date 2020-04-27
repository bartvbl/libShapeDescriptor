#include <omp.h>
#include <spinImage/utilities/fileutils.h>
#include <spinImage/cpu/types/QUICCIImages.h>
#include <spinImage/utilities/readers/quicciReader.h>
#include <bitset>
#include <set>
#include <mutex>
#include <iostream>
#include "SequentialIndexQueryer.h"

std::pair<float, float> computedWeightedHammingWeights(const QuiccImage &needle) {
    unsigned int queryImageSetBitCount = 0;
    for(unsigned int chunk : needle) {
        queryImageSetBitCount += std::bitset<32>(chunk).count();
    }

    const unsigned int bitsPerImage = spinImageWidthPixels * spinImageWidthPixels;
    unsigned int queryImageUnsetBitCount = bitsPerImage - queryImageSetBitCount;

    // If any count is 0, bump it up to 1
    queryImageSetBitCount = std::max<unsigned int>(queryImageSetBitCount, 1);
    queryImageUnsetBitCount = std::max<unsigned int>(queryImageUnsetBitCount, 1);

    // The fewer bits exist of a specific pixel type, the greater the penalty for not containing it
    float missedSetBitPenalty = float(bitsPerImage) / float(queryImageSetBitCount);
    float missedUnsetBitPenalty = float(bitsPerImage) / float(queryImageUnsetBitCount);

    return {missedSetBitPenalty, missedUnsetBitPenalty};
}

float computeWeightedHammingDistance(const QuiccImage &needle, const QuiccImage &haystack, float missedSetBitPenalty, float missedUnsetBitPenalty) {
    // Wherever pixels don't match, we apply a penalty for each of them
    float score = 0;
    for(int i = 0; i < needle.size(); i++) {
        unsigned int wrongSetBitCount = std::bitset<32>((needle[i] ^ haystack[i]) & needle[i]).count();
        unsigned int wrongUnsetBitCount = std::bitset<32>((~needle[i] ^ ~haystack[i]) & ~needle[i]).count();
        score += float(wrongSetBitCount) * missedSetBitPenalty + float(wrongUnsetBitCount) * missedUnsetBitPenalty;
    }

    return score;
}

std::vector<SpinImage::index::QueryResult> SpinImage::index::sequentialQuery(std::experimental::filesystem::path dumpDirectory, const QuiccImage &queryImage, unsigned int resultCount, unsigned int fileStartIndex, unsigned int fileEndIndex) {
    std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();

    std::cout << "Listing files.." << std::endl;
    std::vector<std::experimental::filesystem::path> filesToIndex = SpinImage::utilities::listDirectory(dumpDirectory);
    std::cout << "\tFound " << filesToIndex.size() << " files." << std::endl;

    std::pair<float, float> weights = computedWeightedHammingWeights(queryImage);
    float missedSetBitPenalty = weights.first;
    float missedUnsetBitPenalty = weights.second;

    omp_set_nested(1);
    std::mutex searchResultLock;


    std::set<SpinImage::index::QueryResult> searchResults;
    float currentScoreThreshold = std::numeric_limits<float>::max();

    #pragma omp parallel for schedule(dynamic)
    for (unsigned int fileIndex = fileStartIndex; fileIndex < fileEndIndex; fileIndex++) {

        // Reading image dump file
        std::experimental::filesystem::path archivePath = filesToIndex.at(fileIndex);
        SpinImage::cpu::QUICCIImages images = SpinImage::read::QUICCImagesFromDumpFile(archivePath);

        #pragma omp critical
        {
            // For each image, register pixels in dump file
            #pragma omp parallel for schedule(dynamic)
            for (IndexImageID imageIndex = 0; imageIndex < images.imageCount; imageIndex++) {
                QuiccImage combinedImage = combineQuiccImages(
                        images.horizontallyIncreasingImages[imageIndex],
                        images.horizontallyDecreasingImages[imageIndex]);
                float distanceScore = computeWeightedHammingDistance(queryImage, combinedImage, missedSetBitPenalty, missedUnsetBitPenalty);
                if(distanceScore < currentScoreThreshold || searchResults.size() < resultCount) {
                    searchResultLock.lock();
                    IndexEntry entry = {fileIndex, imageIndex};
                    searchResults.insert({entry, distanceScore, combinedImage});
                    if(searchResults.size() > resultCount) {
                        // Remove worst search result
                        searchResults.erase(std::prev(searchResults.end()));
                        // Update score threshold
                        currentScoreThreshold = std::prev(searchResults.end())->score;
                    }
                    searchResultLock.unlock();
                }
            }

            std::cout << "\rProcessing of file " << fileIndex + 1 << "/" << fileEndIndex << " complete. Current best score: " << currentScoreThreshold << "            " << std::flush;
        }
    }

    std::chrono::steady_clock::time_point endTime = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << std::endl << "Query complete. " << std::endl;
    std::cout << "Total execution time: " << float(duration.count()) / 1000.0f << " seconds" << std::endl;

    std::vector<SpinImage::index::QueryResult> results(searchResults.begin(), searchResults.end());

    for(int i = 0; i < resultCount; i++) {
        std::cout << "Result " << i
                  << ": score " << results.at(i).score
                  << ", file " << results.at(i).entry.fileIndex
                  << ", image " << results.at(i).entry.imageIndex << std::endl;
    }

    return results;
}