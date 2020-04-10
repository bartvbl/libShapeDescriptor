#pragma once

#include <spinImage/cpu/types/QuiccImage.h>
#include <vector>

namespace SpinImage {
    namespace index {
        namespace pattern {
            inline unsigned int computeBitMask(const unsigned int col) {
                return 0x1U << (31U - (col % 32));
            }

            inline unsigned int computeChunkIndex(const unsigned int row, const unsigned int col) {
                return 2 * row + (col / 32);
            }

            inline unsigned int pixelAt(const QuiccImage &image, const unsigned int row, const unsigned int col) {
                unsigned int chunkIndex = computeChunkIndex(row, col);
                unsigned int bitIndex = col % 32;
                return (unsigned int) ((image.at(chunkIndex) >> (31U - bitIndex)) & 0x1U);
            }

            inline bool findNext(
                    QuiccImage &image, QuiccImage &foundPattern,
                    unsigned int &foundPatternSize,
                    unsigned int &row, unsigned int &col,
                    std::vector<std::pair<unsigned short, unsigned short>> &floodFillBuffer) {
                QuiccImage floodFillImage;
                for (; row < 64; row++) {
                    for (; col < 64; col++) {
                        unsigned int pixel = pixelAt(image, row, col);
                        if (pixel == 1) {
                            foundPatternSize = 0;
                            floodFillBuffer.clear();
                            floodFillBuffer.emplace_back(row, col);
                            std::fill(foundPattern.begin(), foundPattern.end(), 0);
                            std::fill(floodFillImage.begin(), floodFillImage.end(), 0);

                            while (!floodFillBuffer.empty()) {
                                std::pair<unsigned short, unsigned short> pixelIndex = floodFillBuffer.at(floodFillBuffer.size() - 1);
                                floodFillBuffer.erase(floodFillBuffer.begin() + floodFillBuffer.size() - 1);
                                int floodFillRow = pixelIndex.first;
                                int floodFillCol = pixelIndex.second;

                                unsigned int floodPixel = pixelAt(image, floodFillRow, floodFillCol);

                                if (floodPixel == 1) {
                                    foundPatternSize++;
                                    // Add pixel to pattern image
                                    unsigned int bitEnablingMask = computeBitMask(floodFillCol);
                                    unsigned int chunkIndex = computeChunkIndex(floodFillRow, floodFillCol);
                                    foundPattern.at(chunkIndex) |= bitEnablingMask;
                                    // Disable pixel
                                    unsigned int bitDisablingMask = ~bitEnablingMask;
                                    image.at(chunkIndex) &= bitDisablingMask;
                                    // Queue surrounding pixels
                                    const int range = 3;
                                    for (int neighbourRow = std::max(int(floodFillRow) - range, 0);
                                         neighbourRow <= std::min(spinImageWidthPixels - 1, floodFillRow + range);
                                         neighbourRow++) {
                                        for (int neighbourCol = std::max(int(floodFillCol) - range, 0);
                                             neighbourCol <= std::min(spinImageWidthPixels - 1, floodFillCol + range);
                                             neighbourCol++) {
                                            unsigned int pixelWasAlreadyVisited = pixelAt(floodFillImage, neighbourRow, neighbourCol);
                                            if(pixelWasAlreadyVisited == 0) {
                                                // Mark the pixel as visited
                                                unsigned int childMarkMask = computeBitMask(neighbourCol);
                                                floodFillImage.at(computeChunkIndex(neighbourRow, neighbourCol)) |= childMarkMask;
                                                floodFillBuffer.emplace_back(neighbourRow, neighbourCol);
                                            }
                                        }
                                    }
                                }
                            }

                            // We have located the pattern, and thus return
                            return true;
                        }
                    }
                    col = 0;
                }

                // No more patterns exist
                return false;
            }
        }
    }
}