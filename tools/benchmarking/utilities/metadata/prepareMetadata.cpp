#include "prepareMetadata.h"
#include <benchmarking/utilities/metadata/generateFakeMetadata.h>
#include <vector>
#include <variant>
#include <string>
#include <filesystem>
#include <fstream>

std::vector<std::variant<int, std::string>> generateMetadata(std::filesystem::path metadataPath)
{
    std::vector<std::variant<int, std::string>> metadata;
    std::ifstream metadataFile;
    std::string line;

    metadataFile.open(metadataPath);
    if (metadataFile.is_open())
    {
        while (getline(metadataFile, line))
        {
            try
            {
                metadata.push_back(stoi(line));
            }
            catch (std::exception e)
            {
                metadata.push_back(line);
            }
        }
        metadataFile.close();
    }

    return metadata;
}

std::vector<std::variant<int, std::string>> Benchmarking::utilities::metadata::prepareMetadata(std::filesystem::path metadataPath, int length)
{
    if (std::filesystem::exists(metadataPath))
    {
        return generateMetadata(metadataPath);
    }

    return Benchmarking::utilities::metadata::generateFakeMetadata(length);
}