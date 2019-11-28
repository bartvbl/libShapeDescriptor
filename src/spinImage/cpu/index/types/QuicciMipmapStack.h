#pragma once

#include <spinImage/libraryBuildSettings.h>

struct QuicciMipmapStack {

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

    QuicciMipmapStack(unsigned int* quiccImage) {
        static_assert(spinImageWidthPixels == 64);
        const int uintsPerRow = spinImageWidthPixels / 2;

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
};