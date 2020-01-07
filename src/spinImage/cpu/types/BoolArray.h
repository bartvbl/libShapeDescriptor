#pragma once

#include <cstddef>
#include <bitset>

template<size_t length> class BoolArray {
private:
    unsigned int arrayContents[(length / 32) + (length % 32 > 0 ? 1 : 0)];

public:
    class BoolReference {
    private:
        const unsigned int* referenceToBitVector;
        const unsigned char bitOffset;

        unsigned int genBitMask(unsigned char offset) {
            return (0x1U << (31U - offset));
        }
    public:
        BoolReference();

        BoolReference(const unsigned int* reference, const unsigned char offset)
        {
            referenceToBitVector = reference;
            bitOffset = offset;
        }

        // array[i] = x;
        BoolReference& operator= (bool value)
        {
            if (value) {
                *referenceToBitVector |= genBitMask(bitOffset);
            } else {
                *referenceToBitVector &= ~genBitMask(bitOffset);
            }
            return *this;
        }

        // array[i] = array[j];
        BoolReference& operator= (const BoolReference& j)
        {
            if ((*(j.referenceToBitVector) & genBitMask(j.bitOffset))) {
                *referenceToBitVector |= genBitMask(j.bitOffset);
            } else {
                *referenceToBitVector &= ~genBitMask(bitOffset);
            }
            return *this;
        }

        // x = array[i];
        operator bool() const
        {
            return (*(referenceToBitVector) & genBitMask(bitOffset)) != 0;
        }
    };

    BoolReference &operator[](size_t index) {
        return BoolReference(arrayContents + (index / 32), index % 32);
    }

};