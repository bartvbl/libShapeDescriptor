#pragma once

#include <spinImage/libraryBuildSettings.h>
#include <iostream>
#include <array>
#include <bitset>
#include <spinImage/cpu/types/QuiccImage.h>
#include "BitSequence.h"

static_assert(spinImageWidthPixels <= 512, "The bit count mipmap stack stores its values as unsigned short, which is able to represent bit counts for images up to 512x512 in size. If you go over this value, make sure to swap out unsigned shorts for unsigned ints.");

struct BitCountMipmapStack {
    //   level   mipmap size    pixel count   area per pixel   value range   space needed
    //   1       8x8 images     64            8x8 pixels       0-64          64 bytes, 8 (6) bits/pixel
    //   2       16x16 images   256           4x4 pixels       0-16          128 bytes, 4 bits/pixel
    //   3       32x32 images   1024          2x2 pixels       0-4           256 bytes, 2 bits/pixel
    //   -- 64x64: source --

    const std::array<unsigned short, 32*32> level6;
    const std::array<unsigned short, 16*16> level5;
    const std::array<unsigned short, 8*8> level4;
    const std::array<unsigned short, 4*4> level3;
    const std::array<unsigned short, 4*2> level2;
    const std::array<unsigned short, 2*2> level1;

    template<unsigned int width, unsigned int height> unsigned long computeSingleBitSequence(const std::array<unsigned short, width*height> &image, std::array<unsigned short, width*height> &mins, std::array<unsigned short, width*height> &maxes) {
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

    BitSequence computeBitSequence() {
        const unsigned short initialMax = (spinImageWidthPixels * spinImageWidthPixels) / 4;
        std::array<unsigned short, 8> mins = {0, 0, 0, 0, 0, 0, 0, 0};
        std::array<unsigned short, 8> maxes = {initialMax, initialMax, initialMax, initialMax, initialMax, initialMax, initialMax, initialMax};
        std::array<unsigned long, 8> bitSequence = {0, 0, 0, 0, 0, 0, 0, 0};

        for(int i = 0; i < 8; i++) {
            computeSingleBitSequence<2, 4>(level2, mins, maxes);
        }

        return {bitSequence};
    }

    std::array<unsigned short, 4 * 2> computeLevel2(const std::array<unsigned short, 4 * 4> level3) {
        std::array<unsigned short, 4 * 2> image;
        for(int row = 0; row < 4; row++) {
            for(int col = 0; col < 2; col++) {
                image[2 * row + col] = level3[4 * row + 2 * col + 0] + level3[4 * row + 2 * col + 1];
            }
        }
        return image;
    }

    std::array<unsigned short, 32*32> computeLevel6(const QuiccImage &quiccImage) {
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

                unsigned int relativeCol = (2 * mipmapCol) % 32;

                image[mipmapRow * 32 + mipmapCol] =
                    int((top >> (31 - relativeCol)) & 0x1) +
                    int((top >> (31 - relativeCol - 1)) & 0x1) +
                    int((bottom >> (31 - relativeCol)) & 0x1) +
                    int((bottom >> (31 - relativeCol - 1)) & 0x1);
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
        : level6(computeLevel6(quiccImage)),
          level5(computeLevel<32>(level6)),
          level4(computeLevel<16>(level5)),
          level3(computeLevel<8>(level4)),
          level2(computeLevel2(level3)),
          level1(computeLevel<4>(level3)) {
        static_assert(spinImageWidthPixels == 64, "The index implementation of the library has been constructed for images of size 64x64");
        //print();
    }

    template<int width, int height> void printLevel(const std::array<unsigned short, width * height> &image) {
        // Origin is in the bottom left corner, but we start printing at the top
        // We thus need to flip the image.
        for(int row = height - 1; row >= 0; row--) {
            std::cout << "\t";
            for(unsigned int col = 0; col < width; col++) {
                int value = (image[row * width + col]);
                std::cout << (value == 0 ? "." : std::to_string(value)) << " ";
            }
            std::cout << std::endl;
        }
    }

    void print() {
        std::cout << std::endl << "Level 1" << std::endl;
        printLevel<2, 2>(level1);

        std::cout << std::endl << "Level 2" << std::endl;
        printLevel<2, 4>(level2);

        std::cout << std::endl << "Level 3" << std::endl;
        printLevel<4, 4>(level3);

        std::cout << std::endl << "Level 4" << std::endl;
        printLevel<8, 8>(level4);

        std::cout << std::endl << "Level 5" << std::endl;
        printLevel<16, 16>(level5);

        std::cout << std::endl << "Level 6" << std::endl;
        printLevel<32, 32>(level6);
    }
};