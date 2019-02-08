#pragma once

#include <string>

struct OutputImageSettings {
    bool enableOutputImage;
    bool enableLogImage;
    std::string imageDestinationFile;
    std::string compressedDestinationFile;
};