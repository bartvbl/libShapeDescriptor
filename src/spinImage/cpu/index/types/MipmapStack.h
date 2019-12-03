#pragma once

#include <spinImage/libraryBuildSettings.h>
#include <iostream>

constexpr int uintsPerRow = spinImageWidthPixels / 32;

// bits per pixel:                     2 bits      4 bits      8 bits      16 bits     32 bits
const unsigned int collatingShiftBitMasks[5] = {0x55555555, 0x11111111, 0x01010101, 0x00010001, 0x00000001};

// bits per pixel:                                   2 -> 4      4 -> 8      8 -> 16     16 -> 32
const unsigned int groupingShiftBitMasks_2bits[4] = {0x33333333, 0x0F0F0F0F, 0x00FF00FF, 0x0000FFFF};
const unsigned int groupingShiftBitMasks_4bits[3] = {            0x03030303, 0x000F000F, 0x000000FF};
const unsigned int groupingShiftBitMasks_8bits[2] = {                        0x00030003, 0x0000000F};
const unsigned int groupingShiftBitMasks_16bits[1]= {                                    0x00000003};

struct MipmapStack {
    //   level   mipmap size    pixel count   area per pixel   value range   space needed
    //   0       4x4 images     16            16x16 pixels     0-256         16 bytes, 8 bits/pixel
    //   1       8x8 images     64            8x8 pixels       0-64          64 bytes, 8 (6) bits/pixel
    //   2       16x16 images   256           4x4 pixels       0-16          128 bytes, 4 bits/pixel
    //   3       32x32 images   1024          2x2 pixels       0-4           256 bytes, 2 bits/pixel
    //   -- 64x64: source --

    //   level   mipmap size    pixel count   space needed
    //   0       4x4 images     16            1 unsigned short
    //   1       8x8 images     64            2 unsigned ints
    //   2       16x16 images   256           8 unsigned ints
    //   3       32x32 images   1024          32 unsigned ints
    //   -- 64x64: source --

    unsigned short level0;
    unsigned int level1[2];
    unsigned int level2[8];
    unsigned int level3[32];

    inline unsigned int compressChunk(
            unsigned int chunk,
            const unsigned int bitsPerPixel,
            const unsigned int* groupingBitShiftMasks,
            const unsigned int initialBitGroupingShiftDistance) {
        // First we shift and OR bits together to collect bits together
        int shiftIndex = 0;
        for(unsigned int shiftDistance = 1; shiftDistance < bitsPerPixel; shiftDistance *= 2) {
            chunk = (chunk | (chunk >> shiftDistance)) & collatingShiftBitMasks[shiftIndex];
            //             ^ The operation applied for collecting the bits
            shiftIndex++;
        }
        // Next we shift all bits to the end, so they are all collected together.
        // Note that the final bit is always already in position, hence we start with the second-to-last
        shiftIndex = 0;
        unsigned int bitGroupingShiftDistance = initialBitGroupingShiftDistance;
        for(unsigned int shiftDistance = bitsPerPixel; shiftDistance < 32; shiftDistance *= 2) {
            chunk = (chunk | (chunk >> bitGroupingShiftDistance)) & groupingBitShiftMasks[shiftIndex];
            shiftIndex++;
            bitGroupingShiftDistance *= 2;
        }
        return chunk;
    }

    void computeMipmapLevel(
            unsigned int* mipmapLevelImage,
            unsigned int *quiccImage,
            const unsigned int mipmapBlockSizeBits,
            const unsigned int* groupingBitShiftMasks,
            const unsigned int initialBitGroupingShiftDistance) {
        unsigned int mipmapChunkIndex = 0;

        const unsigned int bitsPerMipmapRow = spinImageWidthPixels / mipmapBlockSizeBits;
        const unsigned int totalBitCount = bitsPerMipmapRow * bitsPerMipmapRow;
        const unsigned int bitsPerUint = 32 / mipmapBlockSizeBits;

        unsigned int currentChunk = 0;
        unsigned int accumulatedBitCount = 0;
        for(unsigned int rowChunk = 0; rowChunk < spinImageWidthPixels / mipmapBlockSizeBits; rowChunk++) {
            for(unsigned int col = 0; col < uintsPerRow; col++) {
                unsigned int collatedRows = 0;

                for(unsigned int row = 0; row < mipmapBlockSizeBits; row++) {
                    unsigned int chunk = quiccImage[(rowChunk * mipmapBlockSizeBits + row) * uintsPerRow + col];
                    unsigned int shiftIndex = 0;
                    for(unsigned int shiftDistance = 1; shiftDistance < mipmapBlockSizeBits; shiftDistance *= 2) {
                        // Horizontally merge bits together that belong to the same mipmap pixel/block
                        chunk = (chunk | (chunk >> shiftDistance)) & collatingShiftBitMasks[shiftIndex];
                        shiftIndex++;
                    }
                    // Vertically merge bits together
                    collatedRows = collatedRows | chunk;
                }

                // Bits are still distanced apart.
                // We first need to bring them together before we can shift them in place
                unsigned int compressedChunk = compressChunk(collatedRows, mipmapBlockSizeBits, groupingBitShiftMasks, initialBitGroupingShiftDistance);

                currentChunk = currentChunk | (compressedChunk << (32 - bitsPerUint - accumulatedBitCount));
                accumulatedBitCount += bitsPerUint;

                if(accumulatedBitCount == 32) {
                    mipmapLevelImage[mipmapChunkIndex] = currentChunk;
                    mipmapChunkIndex++;

                    // Reset the chunk accumulator
                    accumulatedBitCount = 0;
                    currentChunk = 0;
                }
            }
            // HACK: for computing level 0, we're dealing with a single unsigned short.
            if(totalBitCount == 16) {
                ((unsigned short*) mipmapLevelImage)[mipmapChunkIndex] = (unsigned short) (currentChunk >> 16);
            }
        }
    }

    MipmapStack(unsigned int* quiccImage) {
        static_assert(spinImageWidthPixels == 64);

        // While a giant hack, the computeMipmapLevel() function will detect the case where it deals with
        // an unsigned short as input, and will account for it. In the interest of reusing code this seemed
        // the best solution.

        computeMipmapLevel((unsigned int*) &level0, quiccImage, 16, groupingShiftBitMasks_16bits, 15);
        computeMipmapLevel(level1, quiccImage, 8, groupingShiftBitMasks_8bits, 7);
        computeMipmapLevel(level2, quiccImage, 4, groupingShiftBitMasks_4bits, 3);
        computeMipmapLevel(level3, quiccImage, 2, groupingShiftBitMasks_2bits, 1);
    }

    template<typename printedType> void printBitwiseImage(printedType* image, int size) {
        unsigned int bitIndex = 0;
        unsigned int byteIndex = 0;
        const unsigned int bitsPerType = sizeof(printedType) * 8;
        for(int row = 0; row < size; row++) {
            for(int col = 0; col < size; col++) {
                printedType currentBits = image[byteIndex];
                std::cout << ((currentBits >> (bitsPerType - 1 - bitIndex)) & 0x1);
                bitIndex++;
                if(bitIndex == bitsPerType) {
                    byteIndex++;
                    bitIndex = 0;
                }
            }
            std::cout << std::endl;
        }
    }

    void print() {
        std::cout << "Level 0" << std::endl;
        printBitwiseImage<unsigned short>(&level0, 4);

        std::cout << std::endl << "Level 1" << std::endl;
        printBitwiseImage<unsigned int>(level1, 8);

        std::cout << std::endl << "Level 2" << std::endl;
        printBitwiseImage<unsigned int>(level2, 16);

        std::cout << std::endl << "Level 3" << std::endl;
        printBitwiseImage<unsigned int>(level3, 32);
    }
};