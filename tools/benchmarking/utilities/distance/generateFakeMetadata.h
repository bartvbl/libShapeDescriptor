#pragma once
#include <vector>

namespace Benchmarking
{
    namespace utilities
    {
        namespace distance
        {
            std::vector<std::variant<int, std::string>> generateFakeMetadata(int length);
        }
    }
}