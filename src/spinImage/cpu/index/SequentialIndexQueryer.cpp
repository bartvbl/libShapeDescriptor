#include <omp.h>
#include <spinImage/utilities/fileutils.h>
#include <spinImage/cpu/types/QUICCIImages.h>
#include <spinImage/utilities/readers/quicciReader.h>
#include <bitset>
#include <set>
#include <mutex>
#include <iostream>
#include "SequentialIndexQueryer.h"

float computeWeightedHammingDistance(const QuiccImage &needle, const QuiccImage &haystack) {
    // Wherever pixels don't match, we apply a penalty for each of them
    float score = 0;
    for(int i = 0; i < needle.size(); i++) {
        score += std::bitset<32>(needle[i] ^ haystack[i]).count();
    }

    return score;
}

std::vector<SpinImage::index::QueryResult> SpinImage::index::sequentialQuery(std::experimental::filesystem::path dumpDirectory, const QuiccImage &queryImage, unsigned int resultCount, unsigned int fileStartIndex, unsigned int fileEndIndex, unsigned int threadCount, debug::QueryRunInfo* runInfo) {
    std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();

    std::cout << "Listing files.." << std::endl;
    std::vector<std::experimental::filesystem::path> filesToIndex = SpinImage::utilities::listDirectory(dumpDirectory);
    std::cout << "\tFound " << filesToIndex.size() << " files." << std::endl;

    omp_set_nested(1);
    std::mutex searchResultLock;

    if(threadCount == 0) {
        #pragma omp parallel
        {
            threadCount = omp_get_num_threads();
        };
    }

    std::set<SpinImage::index::QueryResult> searchResults;
    float currentScoreThreshold = std::numeric_limits<float>::max();

    #pragma omp parallel for schedule(dynamic) num_threads(threadCount)
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
                float distanceScore = computeWeightedHammingDistance(queryImage, combinedImage);
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

    /*for(int i = 0; i < resultCount; i++) {
        std::cout << "Result " << i
                  << ": score " << results.at(i).score
                  << ", file " << results.at(i).entry.fileIndex
                  << ", image " << results.at(i).entry.imageIndex << std::endl;
    }*/

    if(runInfo != nullptr) {
        double queryTime = double(duration.count()) / 1000.0;
        runInfo->totalQueryTime = queryTime;
        runInfo->threadCount = threadCount;
        std::fill(runInfo->distanceTimes.begin(), runInfo->distanceTimes.end(), queryTime);
    }

    return results;
}