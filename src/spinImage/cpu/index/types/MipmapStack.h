#pragma once

#include <spinImage/libraryBuildSettings.h>
#include <iostream>

constexpr int uintsPerRow = spinImageWidthPixels / 32;

struct MipmapStack {
    

    //   level   mipmap size    pixel count   area per pixel   value range   space needed
    //   0       4x4 images     16            16x16 pixels     0-256         16 bytes, 8 bits/pixel
    //   1       8x8 images     64            8x8 pixels       0-64          64 bytes, 8 (6) bits/pixel
    //   2       16x16 images   256           4x4 pixels       0-16          128 bytes, 4 bits/pixel
    //   3       32x32 images   1024          2x2 pixels       0-4           256 bytes, 2 bits/pixel
    //   -- 64x64: source --

    unsigned int level0[4];
    unsigned int level1[16];
    unsigned int level2[32];
    unsigned int level3[64];

    void computeMipmapLevel(unsigned int* levelImage, unsigned int *quiccImage, const int quiccPixelsCovered, const int bitsPerMipmapPixel, const unsigned int bitMask) {
        unsigned int mipmapChunkIndex = 0;
        for(unsigned int rowChunk = 0; rowChunk < spinImageWidthPixels / quiccPixelsCovered; rowChunk++) {
            unsigned int currentChunk = 0;
            for(unsigned int col = 0; col < uintsPerRow; col++) {
                unsigned int bitSums = 0;
                unsigned int allSetToOne = bitMask;
                for(unsigned int row = 0; row < quiccPixelsCovered; row++) {
                    unsigned int chunk = quiccImage[(rowChunk * quiccPixelsCovered + row) * uintsPerRow + col];
                    unsigned int partialSum = 0;
                    for(unsigned int bit = 0; bit < quiccPixelsCovered; bit++) {
                        unsigned int filteredBits = (chunk >> bit) & bitMask;
                        allSetToOne = allSetToOne & filteredBits;
                        partialSum += filteredBits;
                    }
                    // Check if last row
                    if(row == quiccPixelsCovered - 1) {
                        // Deduct one from all partial sums where all bits were 1 to avoid overflow
                        // Since all bits are 1, bitSums accumulated at least 1, thus underflow is not possible
                        bitSums -= allSetToOne;
                    }
                    bitSums += partialSum;
                }
                if(quiccPixelsCovered == bitsPerMipmapPixel) {
                    // Write output chunk
                    levelImage[mipmapChunkIndex] = bitSums;
                    mipmapChunkIndex++;
                } else {
                    // This is only true for mipmap level 0
                    // We first move the 16-bit chunk together
                    unsigned int translatedBitSums = ((bitSums & 0x00FF0000) >> 8) | (bitSums & 0x000000FF);
                    // If col is 0, we use the first 16 bits, otherwise we use the second 16 bits
                    currentChunk = currentChunk | (translatedBitSums << ((1 - col) * 16));

                    // If col is 1, the 32-bit chunk is complete, so we can write it.
                    if(col == 1) {
                        // Write output chunk
                        levelImage[mipmapChunkIndex] = currentChunk;
                        mipmapChunkIndex++;
                    }
                }
            }
        }
    }

    MipmapStack(unsigned int* quiccImage) {
        static_assert(spinImageWidthPixels == 64);

        computeMipmapLevel(level0, quiccImage, 16, 8, 0x00010001);
        computeMipmapLevel(level1, quiccImage, 8,  8, 0x01010101);
        computeMipmapLevel(level2, quiccImage, 4,  4, 0x11111111);

        // Level 3
        unsigned int imageOffset = 0;
        for(int row = 0; row < spinImageWidthPixels; row += 2) {
            for(int col = 0; col < uintsPerRow; col++) {
                unsigned int topChunk = quiccImage[row * uintsPerRow + col];
                unsigned int bottomChunk = quiccImage[(row + 1) * uintsPerRow + col];

                unsigned int topChunkHasSingleBit =
                        (topChunk | (topChunk >> 1)) & 0x55555555;
                unsigned int bottomChunkHasSingleBit =
                        (bottomChunk | (bottomChunk >> 1)) & 0x55555555;
                unsigned int singleBitChunksCombined = topChunkHasSingleBit + bottomChunkHasSingleBit;

                unsigned int topChunkHasDoubleBit =
                        (topChunk & (topChunk >> 1)) & 0x55555555;
                unsigned int bottomChunkHasDoubleBit =
                        (bottomChunk & (bottomChunk >> 1)) & 0x55555555;
                unsigned int doubleBitChunksCombined = topChunkHasDoubleBit | bottomChunkHasDoubleBit;

                unsigned int compressedChunk = singleBitChunksCombined + doubleBitChunksCombined;
                level3[imageOffset] = compressedChunk;
                imageOffset++;
            }
        }
    }

    void print() {
        std::cout << "Level 0" << std::endl;
        for(int row = 0; row < 4; row++) {
            std::cout << "\t";
            for(int col = 0; col < 4; col++) {
                std::cout << ((level0[row] >> (24 - 8*col)) & 0xFF) << (col < 3 ? ", " : "");
            }
            std::cout << std::endl;
        }

        std::cout << std::endl << "Level 1" << std::endl;
        for(int row = 0; row < 8; row++) {
            std::cout << "\t";
            for(int col = 0; col < 8; col++) {
                std::cout << ((level1[uintsPerRow * row + col/4] >> (24 - 8*col)) & 0xFF) << (col < 7 ? ", " : "");
            }
            std::cout << std::endl;
        }

        std::cout << std::endl << "Level 2" << std::endl;
        for(int row = 0; row < 16; row++) {
            std::cout << "\t";
            for(int col = 0; col < 16; col++) {
                std::cout << ((level2[uintsPerRow * row + col/8] >> (28 - 4*col)) & 0xF) << (col < 15 ? ", " : "");
            }
            std::cout << std::endl;
        }

        std::cout << std::endl << "Level 3" << std::endl;
        for(int row = 0; row < 32; row++) {
            std::cout << "\t";
            for(int col = 0; col < 32; col++) {
                std::cout << ((level3[uintsPerRow * row + col/16] >> (30 - 2*col)) & 0x3) << (col < 31 ? ", " : "");
            }
            std::cout << std::endl;
        }
    }
};