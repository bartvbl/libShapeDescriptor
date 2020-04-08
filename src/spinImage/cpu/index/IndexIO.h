#pragma once

#include <spinImage/cpu/index/types/Index.h>
#include <spinImage/cpu/index/types/PatternBlockEntry.h>
#include <string>
#include <fstream>


namespace SpinImage {
    namespace index {
        namespace io {
            // Block size is smallest number of cache lines that fit
            // a whole number of image references (assuming the 10 byte size doesn't change)
            const unsigned int BLOCK_SIZE = 32;

            struct PatternBlockFileHandle {
                bool isReplacingFinalBlock = false;

                // Important: array must be initialised before the output stream is initialised
                std::array<PatternBlockEntry, BLOCK_SIZE> currentPatternBlock;

                std::fstream outFile;

                std::experimental::filesystem::path outFileLocation;

                std::fstream initialiseOutFile(const std::experimental::filesystem::path &fileLocation) {
                    if(!std::experimental::filesystem::exists(fileLocation)) {
                        // Create file, and header with dummy information
                        std::fstream outStream(fileLocation, std::ios::binary);

                        // We just keep going afterwards
                        return outStream;
                    } else {
                        // If it does exist, read the file header, and load the last block
                        isReplacingFinalBlock = true;
                        return std::fstream(fileLocation, std::ios::app);
                    }

                }

                PatternBlockFileHandle(const std::experimental::filesystem::path &fileLocation)
                        : outFile(initialiseOutFile(fileLocation)),
                          outFileLocation(fileLocation) {

                }

                void store() {

                }

                void close() {
                    // Write current buffered block to disk
                    outFile.close();
                }
            };

            Index readIndex(std::experimental::filesystem::path indexDirectory);

            void writeIndex(const Index& index, std::experimental::filesystem::path indexDirectory);

            PatternBlockFileHandle* openPatternFile(std::string patternID);
        }
    }
}