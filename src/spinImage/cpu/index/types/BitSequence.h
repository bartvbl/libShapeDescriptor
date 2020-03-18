#pragma once

#include <array>

#define BIT_SEQUENCE_LENGTH 8

class BitSequence {
private:
    const std::array<unsigned long, BIT_SEQUENCE_LENGTH> bitSequence;
public:
    BitSequence(std::array<unsigned long, BIT_SEQUENCE_LENGTH> sequence) : bitSequence(sequence) {}

    bool isBottomLevel(unsigned int level) {
        return level >= BIT_SEQUENCE_LENGTH;
    }

    unsigned long at(unsigned int level) {
        level = std::min<unsigned int>(BIT_SEQUENCE_LENGTH - 1, level);
        return bitSequence.at(level);
    }
};