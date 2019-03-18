#include <fstream>
#include "searchResultDumper.h"

void dumpSearchResults(array<ImageSearchResults> searchResults, size_t imageCount, std::string outputFilePath) {
    std::ofstream outputFile;
    outputFile.open(outputFilePath);

    for(size_t image = 0; image < imageCount; image++) {
        outputFile << "----- Image " << image << " -----" << std::endl;
        outputFile << "Scores: ";
        for (unsigned int i = 0; i < SEARCH_RESULT_COUNT; i++) {
            outputFile << searchResults.content[image].resultScores[i] << (i == SEARCH_RESULT_COUNT-1 ? "\r\n" : ", ");
        }
        outputFile << "Indices: ";
        for (unsigned int i = 0; i < SEARCH_RESULT_COUNT; i++) {
            outputFile << searchResults.content[image].resultIndices[i] << (i == SEARCH_RESULT_COUNT-1 ? "\r\n" : ", ");
        }
        outputFile << "\r\n";
    }

    outputFile.close();
}

void dumpSearchResults(std::vector<std::vector<DescriptorSearchResult>> searchResults, std::string outputFilePath) {
    std::ofstream outputFile;
    outputFile.open(outputFilePath);

    for(size_t image = 0; image < searchResults.size(); image++) {
        size_t resultCount = searchResults.at(image).size();

        outputFile << "----- Image " << image << " -----" << std::endl;
        outputFile << "Scores: ";
        for (unsigned int i = 0; i < resultCount; i++) {
            outputFile << searchResults.at(image).at(i).correlation << (i == resultCount-1 ? "\r\n" : ", ");
        }
        outputFile << "Indices: ";
        for (unsigned int i = 0; i < resultCount; i++) {
            outputFile << searchResults.at(image).at(i).imageIndex << (i == resultCount-1 ? "\r\n" : ", ");
        }
        outputFile << "\r\n";
    }

    outputFile.close();
}