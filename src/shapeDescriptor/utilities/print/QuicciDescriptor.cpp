#include <iostream>
#include "QuicciDescriptor.h"

void ShapeDescriptor::print::quicciDescriptor(ShapeDescriptor::QUICCIDescriptor &descriptor) {
    std::cout << "+";
    for(unsigned int col = 0; col < spinImageWidthPixels; col++) {
        std::cout << "-";
    }
    std::cout << "+" << std::endl;

    for(unsigned int row = 0; row < spinImageWidthPixels; row++) {
        std::cout << '|';
        for(unsigned int col = 0; col < spinImageWidthPixels; col++) {
            unsigned int chunkIndex = (row * spinImageWidthPixels + col) / (8 * sizeof(unsigned int));
            std::cout << (((descriptor.contents[chunkIndex] & (0x80000000U >> (col % 32U))) != 0) ? 'X' : ' ');
        }
        std::cout << '|' << std::endl;
    }

    std::cout << "+";
    for(unsigned int col = 0; col < spinImageWidthPixels; col++) {
        std::cout << "-";
    }
    std::cout << "+" << std::endl;
}
