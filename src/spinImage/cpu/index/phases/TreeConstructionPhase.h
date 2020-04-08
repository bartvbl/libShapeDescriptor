#pragma once

#include <experimental/filesystem>

void constructIndexTree(std::experimental::filesystem::path quicciImageDumpDirectory,
                        std::experimental::filesystem::path indexDumpDirectory,
                        size_t cachedPatternLimit,
                        size_t fileStartIndex, size_t fileEndIndex);