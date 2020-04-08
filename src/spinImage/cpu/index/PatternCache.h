#pragma once

#include <spinImage/utilities/Cache.h>
#include <experimental/filesystem>
#include <fstream>
#include <spinImage/cpu/index/types/PatternBlockEntry.h>

class PatternCache : Cache<std::string, PatternBlockFileHandle> {
    void onEviction(PatternBlockFileHandle* item) {

    }

    void eject(PatternBlockFileHandle* item) {

    }

    PatternBlockFileHandle* load(std::string &itemID) {
        // If the file does not exist, create it.
        // Read the file header. It should contain the number of images present, and a pointer to the start of the final block
        // Read the final block
        // Remove the final block???
        // Put the handle and buffer in memory
    }
};