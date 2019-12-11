#include "BoolVector.h"

void BoolVector::push_back(bool bit) {
    unsigned int bitInUint = length % 32;
    if(bitInUint == 0) {
        container.push_back(0);
    }
    unsigned long uintIndex = length / 32;
    unsigned int bitVector = container.at(uintIndex);
    bitVector |= (bit ? 0x1U : 0x0U) << (31U - bitInUint);
    container.at(uintIndex) = bitVector;

    length++;
}

void BoolVector::reserve(unsigned long size) {
    container.reserve((size / 32) + (size % 32 == 0 ? 0 : 1));
}

bool BoolVector::at(unsigned long index) {
    unsigned int indexInBitVector = index % 32;
    return ((container.at(index / 32U) >> (31U - indexInBitVector)) & 0x1U) == 0x1U;
}

unsigned char *BoolVector::data() {
    return (unsigned char*) container.data();
}

unsigned long BoolVector::sizeInBytes() {
    return container.size() * sizeof(unsigned int);
}

unsigned long BoolVector::size() {
    return length;
}

void BoolVector::resize(unsigned long size) {
    container.resize((size / 32) + (size % 32 == 0 ? 0 : 1));
    length = size;
}
