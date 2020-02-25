#pragma once

#include <cstddef>
#include <bitset>

template<size_t length> class BoolArray {

public:
    static constexpr size_t computeArrayLength() {
        return (length / 32) + (length % 32 > 0 ? 1 : 0);
    }

private:

    const unsigned int genBitMask(unsigned char offset) const {
        return (0x1U << (31U - offset));
    }
    unsigned int arrayContents[computeArrayLength()];

public:

    void set(unsigned int index, bool value)
    {
        unsigned int bitOffset = index % 32;
        unsigned int chunkIndex = index / 32;
        if (value) {
            arrayContents[chunkIndex] |= genBitMask(bitOffset);
        } else {
            arrayContents[chunkIndex] &= ~genBitMask(bitOffset);
        }
    }

    // x = array[i];
    const bool operator[](unsigned int index) const
    {
        unsigned int bitOffset = index % 32;
        unsigned int chunkIndex = index / 32;
        return (arrayContents[chunkIndex] & genBitMask(bitOffset)) != 0;
    }


    BoolArray(bool initialValue) {
        for(size_t i = 0; i < computeArrayLength(); i++) {
            arrayContents[i] = initialValue ? 0xFFFFFFFF : 0x00000000;
        }
    }

    const unsigned int* data() const {
        return arrayContents;
    }

    BoolArray() {}
};