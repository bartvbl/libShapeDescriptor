#pragma once

#include <spinImage/cpu/types/BoolArray.h>
#include <stdexcept>

enum NodeLinkType {
    INDEX_NODE,
    LEAF_NODE,
    BUCKET_NODE
};

template<int length> class NodeLinkArray {
private:
    BoolArray<2 * length> backendArray;
public:
    class NodeLinkAccessor {

        const BoolArray<2 * length>* accessorReference;
        const size_t index;

        NodeLinkAccessor(const BoolArray<2 * length> *arrayReference, const size_t arrayIndex)
                : accessorReference(arrayReference), index(arrayIndex) {}

        // x = array[i]
        explicit operator NodeLinkType() const {
            bool bit1 = (*accessorReference).backendArray[2 * index + 0];
            bool bit2 = (*accessorReference).backendArray[2 * index + 1];

            if(!bit1 && !bit2) {
                return INDEX_NODE;
            } else if(bit1 && !bit2) {
                return LEAF_NODE;
            } else if(!bit1 /* && bit2 */) {
                return BUCKET_NODE;
            } else {
                throw std::runtime_error("Invalid node type detected!");
            }
        }

        // array[i] = x;
        NodeLinkAccessor &operator=(NodeLinkType value) {
            bool bit1;
            bool bit2;

            switch(value) {
                case INDEX_NODE:
                    bit1 = false;
                    bit2 = false;
                    break;
                case LEAF_NODE:
                    bit1 = true;
                    bit2 = false;
                    break;
                case BUCKET_NODE:
                    bit1 = false;
                    bit2 = true;
                    break;
                default:
                    throw std::runtime_error("Unsupported node type detected!");
            }

            (*accessorReference).backendArray[2 * index + 0] = bit1;
            (*accessorReference).backendArray[2 * index + 1] = bit2;

            return *this;
        }
    };

    NodeLinkAccessor &operator[](size_t index) {
        return NodeLinkAccessor(&backendArray, index);
    }


};