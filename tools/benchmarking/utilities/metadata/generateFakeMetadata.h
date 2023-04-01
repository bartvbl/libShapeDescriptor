#pragma once
#include <vector>
#include <variant>
#include <string>

namespace Benchmarking
{
    namespace utilities
    {
        namespace metadata
        {
            std::vector<std::variant<int, std::string>> generateFakeMetadata(int length);
        }
    }
}