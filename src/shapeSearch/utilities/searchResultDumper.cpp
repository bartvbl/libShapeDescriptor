#include <fstream>
#include "searchResultDumper.h"

void dumpSearchResults(array<ImageSearchResults> searchResults, size_t imageCount, std::string outputFilePath) {
    std::ofstream outputFile;
    outputFile.open(outputFilePath);

    for(size_t image = 0; image < imageCount; image++) {
        for (unsigned int i = 0; i < SEARCH_RESULT_COUNT; i++) {
            outputFile << "Scores: " << searchResults.content[image].resultScores[i] << (i == 31 ? "\r\n" : ", ");
            outputFile << "Indices: " <<  searchResults.content[image].resultIndices[i] << (i == 31 ? "\r\n" : ", ");
        }
    }

    outputFile.close();
}