#pragma once

#include "MipmapStack.h"

struct BitwiseMipmapStack {

    //   level   mipmap size    pixel count   space needed
    //   0       4x4 images     16            1 unsigned short     8 bits per pixel
    //   1       8x8 images     64            2 unsigned ints      8 bits per pixel
    //   2       16x16 images   256           8 unsigned ints      4 bits per pixel
    //   3       32x32 images   1024          32 unsigned ints     2 bits per pixel
    //   -- 64x64: source --

    unsigned short level0;
    unsigned int level1[2];
    unsigned int level2[8];
    unsigned int level3[32];

    // bits per pixel:                     2 bits      4 bits      8 bits      16 bits     32 bits
    const unsigned int shiftBitMasks[5] = {0x55555555, 0x11111111, 0x01010101, 0x00010001, 0x00000001};

    inline unsigned int compressChunk(unsigned int chunk, const unsigned int bitsPerPixel) {
        // First we shift and OR bits together to collect bits together
        int shiftIndex = 0;
        for(unsigned int shiftDistance = 1; shiftDistance < bitsPerPixel; shiftDistance = shiftDistance << 1) {
            chunk = (chunk | (chunk >> shiftDistance)) & shiftBitMasks[shiftIndex];
            //             ^ The reduction operator applied for collecting the bits
            shiftIndex++;
        }
        // Next we shift all bits to the end, so they are all collected together.
        // Note that the final bit is always already in position, hence we start with the second-to-last
        unsigned int bitIndex = 1;
        // The first bit must always be copied regardless. We therefore use it to initialise the output compressedChunk
        unsigned int compressedChunk = chunk & 0x1;
        for(unsigned int shiftDistance = bitsPerPixel; shiftDistance < 32; shiftDistance += bitsPerPixel) {
            // Copies one bit at a time selected from chunk into compressedChunk
            compressedChunk = compressedChunk | ((chunk >> (shiftDistance - bitIndex)) & (0x1 << bitIndex));
            bitIndex++;
        }
        return compressedChunk;
    }

    void computeBitwiseMipmap(unsigned int* bitwiseMipmap, unsigned int* mipmap, const unsigned int bitsPerPixel, const unsigned int inputChunkCount, const unsigned int outputChunkCount) {
        const unsigned int inputChunksPerOutputChunk = bitsPerPixel;

        for(unsigned int outputChunkIndex = 0; outputChunkIndex < outputChunkCount; outputChunkIndex++) {
            unsigned int outputChunk = 0;
            // An output chunk (bitwise mipmap) consists of multiple input (regular mipmap) chunks
            // We account separately for the case where we only have 16 bits in total
            for(unsigned int inputChunkIndex = 0; inputChunkIndex < std::min(inputChunkCount, inputChunksPerOutputChunk); inputChunkIndex++) {
                unsigned int inputChunk = mipmap[inputChunkIndex];
                unsigned int compressedChunk = compressChunk(inputChunk, bitsPerPixel);
                outputChunk = outputChunk | compressedChunk << 32 - (inputChunkIndex * bitsPerPixel);
            }
            bitwiseMipmap[outputChunkIndex] = outputChunk;
        }
    }

    BitwiseMipmapStack(MipmapStack sourceStack) {
        unsigned int tempLevel0;
        computeBitwiseMipmap(&tempLevel0, sourceStack.level0, 8, 4, 1);
        level0 = (unsigned short) tempLevel0;
    }
};