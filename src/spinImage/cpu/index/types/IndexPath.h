#pragma once

#include <array>
#include "BitCountMipmapStack.h"

#define INDEX_PATH_MAX_LENGTH 8

class IndexPath {
private:
    const std::array<unsigned long, INDEX_PATH_MAX_LENGTH> bitSequence;

    template<unsigned int width, unsigned int height> unsigned long computeSingleBitSequence(
            const std::array<unsigned short, width*height> &image, 
            std::array<unsigned short, width*height> &mins, 
            std::array<unsigned short, width*height> &maxes) {
        unsigned long bitSequence = 0;

        for(unsigned int i = 0; i < width * height; i++) {
            unsigned short pivot = (maxes[i] - mins[i]) / 2;
            bool directionBit = image[i] >= pivot;
            bitSequence = bitSequence | (((unsigned int) directionBit) << (width * height - 1 - i));
            if(directionBit) {
                mins[i] = pivot;
            } else {
                maxes[i] = pivot;
            }
        }

        return bitSequence;
    }

    std::array<unsigned long, 8> computeBitSequence(BitCountMipmapStack mipmapStack) {
        const unsigned short initialMax = (spinImageWidthPixels * spinImageWidthPixels) / 8;
        std::array<unsigned short, 8> mins = {0, 0, 0, 0, 0, 0, 0, 0};
        std::array<unsigned short, 8> maxes = {initialMax, initialMax, initialMax, initialMax, initialMax, initialMax, initialMax, initialMax};
        std::array<unsigned long, 8> bitSequence = {0, 0, 0, 0, 0, 0, 0, 0};

        for(int i = 0; i < 8; i++) {
            bitSequence.at(i) = computeSingleBitSequence<2, 4>(mipmapStack.level2, mins, maxes);
        }

        return bitSequence;
    }
public:
    BitSequence(BitCountMipmapStack mipmapStack) : bitSequence(computeBitSequence(mipmapStack)) {}

    bool isBottomLevel(unsigned int level) {
        return level >= BIT_SEQUENCE_LENGTH;
    }

    unsigned long at(unsigned int level) {
        level = std::min<unsigned int>(BIT_SEQUENCE_LENGTH - 1, level);
        return bitSequence.at(level);
    }

    unsigned short computeMinDistanceTo(BitSequence otherSequence, unsigned int level) {

    }
};