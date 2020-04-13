#pragma once

#include <experimental/filesystem>

void buildSimpleListIndex(
        const std::experimental::filesystem::path &quicciImageDumpDirectory,
        std::experimental::filesystem::path &indexDumpDirectory,
        size_t cachedPatternLimit,
        size_t fileStartIndex, size_t fileEndIndex);