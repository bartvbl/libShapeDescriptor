#pragma once

#include <spinImage/libraryBuildSettings.h>
#include <iostream>
#include <array>
#include <bitset>
#include <spinImage/cpu/types/QuiccImage.h>

struct BitCountMipmapStack {
    //   level   mipmap size    pixel count   area per pixel   value range   space needed
    //   1       8x8 images     64            8x8 pixels       0-64          64 bytes, 8 (6) bits/pixel
    //   2       16x16 images   256           4x4 pixels       0-16          128 bytes, 4 bits/pixel
    //   3       32x32 images   1024          2x2 pixels       0-4           256 bytes, 2 bits/pixel
    //   -- 64x64: source --

    const std::array<unsigned short, 32*32> level5;
    const std::array<unsigned short, 16*16> level4;
    const std::array<unsigned short, 8*8> level3;
    const std::array<unsigned short, 4*4> level2;
    const std::array<unsigned short, 2*2> level1;

    std::array<unsigned short, 32*32> computeLevel5(const QuiccImage &quiccImage) {
        std::array<unsigned short, 32*32> image;

        for(int mipmapRow = 0; mipmapRow < 32; mipmapRow++) {
            for(int mipmapCol = 0; mipmapCol < 32; mipmapCol++) {
                // One row is 2 unsigned integers
                const unsigned int uintsPerRow = 2;
                unsigned int quiccImageRow = 2 * mipmapRow;
                unsigned int quiccImageCol = mipmapCol / 16;
                unsigned int topIndex = quiccImageRow * uintsPerRow + quiccImageCol;
                unsigned int bottomIndex = (quiccImageRow + 1) * uintsPerRow + quiccImageCol;
                unsigned int top = quiccImage[topIndex];
                unsigned int bottom = quiccImage[bottomIndex];

                image[mipmapRow * 32 + mipmapCol] =
                    int((top >> (31 - mipmapCol)) & 0x1) +
                    int((top >> (31 - mipmapCol - 1)) & 0x1) +
                    int((bottom >> (31 - mipmapCol)) & 0x1) +
                    int((bottom >> (31 - mipmapCol - 1)) & 0x1);
            }
        }

        return image;
    }

    template<int edgeSize> std::array<unsigned short, (edgeSize/2)*(edgeSize/2)> computeLevel(
            const std::array<unsigned short, edgeSize*edgeSize> &inputImage) {
        std::array<unsigned short, (edgeSize/2)*(edgeSize/2)> image;

        for(int mipmapRow = 0; mipmapRow < edgeSize/2; mipmapRow++) {
            for(int mipmapCol = 0; mipmapCol < edgeSize/2; mipmapCol++) {
                image[(mipmapRow * edgeSize/2) + mipmapCol] =
                    inputImage[(edgeSize * 2 * mipmapRow) + (2 * mipmapCol)] +
                    inputImage[(edgeSize * 2 * mipmapRow) + (2 * mipmapCol) + 1] +
                    inputImage[(edgeSize * (2 * mipmapRow + 1)) + (2 * mipmapCol)] +
                    inputImage[(edgeSize * (2 * mipmapRow + 1)) + (2 * mipmapCol) + 1];
            }
        }

        return image;
    }

    explicit BitCountMipmapStack(const QuiccImage &quiccImage)
        : level5(computeLevel5(quiccImage)),
          level4(computeLevel<32>(level5)),
          level3(computeLevel<16>(level4)),
          level2(computeLevel<8>(level3)),
          level1(computeLevel<4>(level2)) {
        static_assert(spinImageWidthPixels == 64, "The index implementation of the library has been constructed for images of size 64x64");
        print();
    }

    template<int edgeSize> void printLevel(const std::array<unsigned short, edgeSize * edgeSize> &image) {
        for(unsigned int row = 0; row < edgeSize; row++) {
            std::cout << "\t";
            for(unsigned int col = 0; col < edgeSize; col++) {
                int value = (image[row * edgeSize + col]);
                std::cout << (value == 0 ? "." : std::to_string(value)) << " ";
            }
            std::cout << std::endl;
        }
    }

    void print() {
        std::cout << std::endl << "Level 1" << std::endl;
        printLevel<2>(level1);

        std::cout << std::endl << "Level 2" << std::endl;
        printLevel<4>(level2);

        std::cout << std::endl << "Level 3" << std::endl;
        printLevel<8>(level3);

        std::cout << std::endl << "Level 4" << std::endl;
        printLevel<16>(level4);

        std::cout << std::endl << "Level 5" << std::endl;
        printLevel<32>(level5);
    }
};