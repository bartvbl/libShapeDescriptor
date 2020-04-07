#include "Pattern.h"

bool SpinImage::index::pattern::findNext(
        QuiccImage &image, QuiccImage &foundPattern,
        unsigned int &foundPatternSize,
        unsigned int &row, unsigned int &col,
        std::vector<std::pair<unsigned short, unsigned short>> &floodFillBuffer) {
    QuiccImage floodFillImage;
    for (; row < 64; row++) {
        for (; col < 64; col++) {
            unsigned int pixel = (unsigned int) ((image.at(2 * row + (col / 32)) >> (31U - col)) & 0x1U);

            if (pixel == 1) {
                foundPatternSize = 0;
                floodFillBuffer.clear();
                floodFillBuffer.emplace_back(row, col);
                std::fill(foundPattern.begin(), foundPattern.end(), 0);
                std::fill(floodFillImage.begin(), floodFillImage.end(), 0);

                while (!floodFillBuffer.empty()) {
                    std::pair<unsigned short, unsigned short> pixelIndex = floodFillBuffer.at(floodFillBuffer.size() - 1);
                    floodFillBuffer.erase(floodFillBuffer.begin() + floodFillBuffer.size() - 1);
                    unsigned int chunkIndex = 2 * pixelIndex.first + (pixelIndex.second / 32);
                    unsigned int chunk = image.at(chunkIndex);
                    unsigned int floodPixel = (unsigned int)
                            ((chunk >> (31U - pixelIndex.second % 32)) & 0x1U);

                    if (floodPixel == 1) {
                        foundPatternSize++;
                        // Add pixel to pattern image
                        unsigned int bitEnablingMask = 0x1U << (31U - pixelIndex.second % 32);
                        foundPattern.at(chunkIndex) |= bitEnablingMask;
                        // Disable pixel
                        unsigned int bitDisablingMask = ~bitEnablingMask;
                        image.at(chunkIndex) = chunk & bitDisablingMask;
                        // Queue surrounding pixels
                        const int range = 3;
                        for (int floodRow = std::max(int(pixelIndex.first) - range, 0);
                             floodRow <= std::min(63, pixelIndex.first + range);
                             floodRow++) {
                            for (int floodCol = std::max(int(pixelIndex.second) - range, 0);
                                 floodCol <= std::min(63, pixelIndex.second + range);
                                 floodCol++) {
                                unsigned int childChunkIndex = 2 * floodRow + (floodCol / 32);
                                unsigned int childChunk = floodFillImage.at(childChunkIndex);
                                unsigned int pixelWasAlreadyVisited = (unsigned int)
                                        ((childChunk >> (31U - floodCol % 32)) & 0x1U);
                                if(pixelWasAlreadyVisited == 0) {
                                    // Mark the pixel as visited
                                    unsigned int childMarkMask = 0x1U << (31U - floodCol % 32);
                                    floodFillImage.at(childChunkIndex) |= childMarkMask;
                                    floodFillBuffer.emplace_back(floodRow, floodCol);
                                }
                            }
                        }
                        // We have located the pattern, and thus return
                        return true;
                    }
                }
            }
        }
    }
    // No more patterns exist
    return false;
}