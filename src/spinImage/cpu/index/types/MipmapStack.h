#pragma once

#include <spinImage/libraryBuildSettings.h>
#include <iostream>

constexpr int uintsPerRow = spinImageWidthPixels / 32;

// bits per pixel:                                   2 -> 4      4 -> 8      8 -> 16     16 -> 32
const unsigned int groupingShiftBitMasks_2bits[4] = {0x33333333, 0x0F0F0F0F, 0x00FF00FF, 0x0000FFFF};

// Shift all bits to the end, so they are all collected together.
inline unsigned int compressChunk_2bits(unsigned int chunk) {
    unsigned int shiftIndex = 0;
    for(unsigned int shiftDistance = 1; shiftDistance < 16; shiftDistance *= 2) {
        chunk = (chunk | (chunk >> shiftDistance)) & groupingShiftBitMasks_2bits[shiftIndex];
        shiftIndex++;
    }
    return chunk;
}

struct MipMapImage {
    unsigned int compare(const MipMapImage other, const unsigned int) {

    }
};

struct MipMapLevel3 {
    const std::array<unsigned int, 32> image;

    // 64x64 -> 32x32 image
    static std::array<unsigned int, 32> computeMipmapLevel3(const unsigned int* quiccImage) {
        std::array<unsigned int, 32> level3;

        for(int row = 0; row < spinImageWidthPixels; row += 2) {
            unsigned int topLeftChunk = quiccImage[row * uintsPerRow + 0];
            unsigned int topRightChunk = quiccImage[row * uintsPerRow + 1];
            unsigned int bottomLeftChunk = quiccImage[(row + 1) * uintsPerRow + 0];
            unsigned int bottomRightChunk = quiccImage[(row + 1) * uintsPerRow + 1];

            unsigned int topLeftCombined = (topLeftChunk | (topLeftChunk >> 1U)) & 0x55555555U;
            unsigned int bottomLeftCombined = (bottomLeftChunk | (bottomLeftChunk >> 1U)) & 0x55555555U;
            unsigned int topRightCombined = (topRightChunk | (topRightChunk >> 1U)) & 0x55555555U;
            unsigned int bottomRightCombined = (bottomRightChunk | (bottomRightChunk >> 1U)) & 0x55555555U;

            unsigned int compressedLeftChunk = compressChunk_2bits(topLeftCombined | bottomLeftCombined);
            unsigned int compressedRightChunk = compressChunk_2bits(topRightCombined | bottomRightCombined);

            level3[row / 2] = (compressedLeftChunk << 16U) | compressedRightChunk;
        }
        return level3;
    }

    MipMapLevel3(unsigned int* quicciImage) : image(computeMipmapLevel3(quicciImage)) {}
};

struct MipMapLevel2 {
    const std::array<unsigned int, 8> image;

    // 32x32 -> 16x16 image
    static std::array<unsigned int, 8> computeMipmapLevel2(MipMapLevel3 level3) {
        unsigned int combinedCompressedChunk = 0;
        std::array<unsigned int, 8> level2;

        for(unsigned int row = 0; row < 32; row += 2) {
            unsigned int topChunk = level3.image[row];
            unsigned int bottomChunk = level3.image[row + 1];

            unsigned int topCombined = (topChunk | (topChunk >> 1U)) & 0x55555555U;
            unsigned int bottomCombined = (bottomChunk | (bottomChunk >> 1U)) & 0x55555555U;

            unsigned int compressedChunk = compressChunk_2bits(topCombined | bottomCombined);

            // We need 2 16 bit compressed chunks to make 1 32-bit output chunk.
            // So we save every other of these chunks, and only write when we need to.
            if((row & 0x2U) == 0) {
                combinedCompressedChunk = (compressedChunk << 16U);
            } else {
                level2[row / 4] = combinedCompressedChunk | compressedChunk;
            }
        }

        return level2;
    }

    MipMapLevel2(MipMapLevel3 level3Image) : image(computeMipmapLevel2(level3Image)) {}
};

struct MipMapLevel1 {
    const std::array<unsigned int, 2> image;

    // 16x16 -> 8x8 image
    static std::array<unsigned int, 2> computeMipmapLevel1(MipMapLevel2 level2) {
        std::array<unsigned int, 2> level1;

        unsigned int combinedCompressedChunk = 0;
        unsigned char byteIndex = 0;

        for(unsigned int chunk = 0; chunk < 8; chunk++) {
            unsigned int doubleRowChunk = level2.image[chunk];

            unsigned int combined = (doubleRowChunk | (doubleRowChunk >> 1U)) & 0x55555555U;
            combined = (combined | (combined >> 16U)) & 0x00005555U;

            unsigned int compressedChunk = compressChunk_2bits(combined);

            // Every 4 chunks produces one output chunk
            // (8 bits per processed chunk)
            combinedCompressedChunk = combinedCompressedChunk | (compressedChunk << (32U - 8U - 8U*byteIndex));
            if(byteIndex == 3) {
                level1[chunk / 4] = combinedCompressedChunk;
                combinedCompressedChunk = 0;
                byteIndex = 0;
            } else {
                byteIndex++;
            }
        }
        return level1;
    }

    MipMapLevel1(MipMapLevel2 higherLevelImage) : image(computeMipmapLevel1(higherLevelImage)) {}
};

struct MipMapLevel0 {
    const unsigned short image;

    static const unsigned short computeMipmapLevel0(MipMapLevel1 level1) {
        unsigned int partialChunks[2] = {0, 0};

        for(int i = 0; i < 2; i++) {
            unsigned int chunk = level1.image[i];
            // Combine pixels horizontally
            chunk = (chunk | (chunk >> 1U)) & 0x55555555U;
            // Combine pixels vertically
            chunk = (chunk | (chunk >> 8U)) & 0x00550055U;
            // Shift pixel bits together
            chunk = (chunk | (chunk >> 8U)) & 0x00005555U;
            // Compress like a regular 2-bit distanced chunk
            // Leaves the final 8 bits populated.
            partialChunks[i] = compressChunk_2bits(chunk);
        }

        // Both chunks produce 8 bits each, so we combine them into an unsigned short here.
        return (unsigned short) ((partialChunks[0] << 8U) | partialChunks[1]);
    }

    MipMapLevel0(MipMapLevel1 level1) : image(computeMipmapLevel0(level1)) {}
};

struct MipmapStack {
    //   level   mipmap size    pixel count   area per pixel   space needed
    //   0       4x4 images     16            16x16 pixels     1 unsigned short
    //   1       8x8 images     64            8x8 pixels       2 unsigned ints
    //   2       16x16 images   256           4x4 pixels       8 unsigned ints
    //   3       32x32 images   1024          2x2 pixels       32 unsigned ints
    //   -- 64x64: source --

    // The order in which these are defined matters due to the initialiser list of the constructor
    MipMapLevel3 level3;
    MipMapLevel2 level2;
    MipMapLevel1 level1;
    MipMapLevel0 level0;

    MipmapStack(unsigned int* quiccImage) :
            level3(quiccImage),
            level2(level3),
            level1(level2),
            level0(level1) {
        static_assert(spinImageWidthPixels == 64);
    }

    template<typename printedType, int intCount> void printBitwiseImage(const std::array<printedType, intCount> &image, int size) {
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
        std::array<unsigned short, 1> tempImage = {level0.image};
        printBitwiseImage<unsigned short, 1>(tempImage, 4);

        std::cout << std::endl << "Level 1" << std::endl;
        printBitwiseImage<unsigned int, 2>(level1.image, 8);

        std::cout << std::endl << "Level 2" << std::endl;
        printBitwiseImage<unsigned int, 8>(level2.image, 16);

        std::cout << std::endl << "Level 3" << std::endl;
        printBitwiseImage<unsigned int, 32>(level3.image, 32);
    }
};