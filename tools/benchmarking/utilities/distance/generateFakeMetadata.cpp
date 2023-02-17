#include "generateFakeMetadata.h"
#include <vector>
#include <variant>
#include <string>

std::vector<std::variant<int, std::string>> Benchmarking::utilities::distance::generateFakeMetadata(int length)
{
    std::vector<std::variant<int, std::string>> fakeMetadata;

    for (int i = 0; i < length; i++)
    {
        fakeMetadata.push_back(i);
    }

    return fakeMetadata;
}