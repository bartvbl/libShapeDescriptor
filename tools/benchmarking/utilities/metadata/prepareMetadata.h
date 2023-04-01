#pragma once
#include <vector>
#include <string>
#include <variant>
#include <shapeDescriptor/cpu/types/array.h>
#include <filesystem>

namespace Benchmarking
{
    namespace utilities
    {
        namespace metadata
        {
            std::vector<std::variant<int, std::string>> prepareMetadata(std::filesystem::path metadataPath, int length = 0);
        }
    }
}