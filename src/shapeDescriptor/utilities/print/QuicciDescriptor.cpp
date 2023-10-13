#include <iostream>
#include <shapeDescriptor/shapeDescriptor.h>

void ShapeDescriptor::printQuicciDescriptor(ShapeDescriptor::QUICCIDescriptor &descriptor) {
    std::cout << "+";
    for(unsigned int col = 0; col < spinImageWidthPixels; col++) {
        std::cout << "-";
    }
    std::cout << "+" << std::endl;

    for(int row = spinImageWidthPixels - 1; row >= 0; row--) {
        std::cout << '|';
        for(unsigned int col = 0; col < spinImageWidthPixels; col++) {
            unsigned int chunkIndex = (((unsigned int) row) * spinImageWidthPixels + col) / (8 * sizeof(unsigned int));
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
