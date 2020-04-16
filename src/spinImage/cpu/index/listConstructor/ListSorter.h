#pragma once

#include <experimental/filesystem>

void sortListFiles(
        const std::experimental::filesystem::path &quicciImageDumpDirectory,
        std::experimental::filesystem::path &indexDumpDirectory,
        size_t openFileLimit,
        size_t fileStartIndex, size_t fileEndIndex);