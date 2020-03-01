#pragma once

#include <spinImage/libraryBuildSettings.h>
#include <iostream>
#include <array>
#include <bitset>
#include <spinImage/cpu/types/QuiccImage.h>

//constexpr int uintsPerRow = spinImageWidthPixels / 32;

struct BitCountMipmapStack {


    //   level   mipmap size    pixel count   area per pixel   value range   space needed
    //   1       8x8 images     64            8x8 pixels       0-64          64 bytes, 8 (6) bits/pixel
    //   2       16x16 images   256           4x4 pixels       0-16          128 bytes, 4 bits/pixel
    //   3       32x32 images   1024          2x2 pixels       0-4           256 bytes, 2 bits/pixel
    //   -- 64x64: source --

    const std::array<unsigned short, 32*32> level3;
    const std::array<unsigned short, 16*16> level2;
    const std::array<unsigned short, 8*8> level1;

    std::array<unsigned short, 32*32> computeLevel3(const QuiccImage &quiccImage) {
        std::array<unsigned short, 32*32> image;

        for(int mipmapRow = 0; mipmapRow < 32; mipmapRow++) {
            for(int mipmapCol = 0; mipmapCol < 32; mipmapCol++) {
                unsigned int top = quiccImage[2 * 2 * mipmapRow + (mipmapCol / 16)];
                unsigned int bottom = quiccImage[2 * 2 * mipmapRow + (mipmapCol / 16) + 2];
                std::bitset<32> topBits(top);
                std::bitset<32> bottomBits(bottom);
                unsigned char relativeCol = (2 * mipmapCol) % 32;

                const char bitShuffleIndices[32] =
                    {24, 25, 26, 27,
                     16, 17, 18, 19,
                     8,  9,  10, 11,
                     0,  1,  2,  3,
                     4,  5,  6,  7,
                     12, 13, 14, 15,
                     20, 21, 22, 23,
                     28, 29, 30, 31};

                int transposedMipmapRow = mipmapCol;
                int transposedMipmapCol = bitShuffleIndices[mipmapRow];

                image[transposedMipmapRow * 32 + transposedMipmapCol] =
                    int(topBits[31 - relativeCol]) +
                    int(topBits[31 - (relativeCol + 1)]) +
                    int(bottomBits[31 - relativeCol]) +
                    int(bottomBits[31 - (relativeCol + 1)]);
            }
        }

        return image;
    }

    std::array<unsigned short, 16*16> computeLevel2() {
        std::array<unsigned short, 16*16> image;

        for(int mipmapRow = 0; mipmapRow < 16; mipmapRow++) {
            for(int mipmapCol = 0; mipmapCol < 16; mipmapCol++) {
                image[mipmapRow * 16 + mipmapCol] =
                    level3[(32 * (2 * mipmapRow) + (2 * mipmapCol))] +
                    level3[(32 * (2 * mipmapRow) + (2 * mipmapCol) + 1)] +
                    level3[(32 * (2 * mipmapRow + 1)) + (2 * mipmapCol)] +
                    level3[(32 * (2 * mipmapRow + 1)) + (2 * mipmapCol) + 1];
            }
        }

        return image;
    }

    std::array<unsigned short, 8*8> computeLevel1() {
        std::array<unsigned short, 8*8> image;

        for(int mipmapRow = 0; mipmapRow < 8; mipmapRow++) {
            for(int mipmapCol = 0; mipmapCol < 8; mipmapCol++) {
                image[mipmapRow * 8 + mipmapCol] =
                    level2[(16 * 2 * mipmapRow) + (2 * mipmapCol)] +
                    level2[(16 * 2 * mipmapRow) + (2 * mipmapCol) + 1] +
                    level2[(16 * (2 * mipmapRow + 1)) + (2 * mipmapCol)] +
                    level2[(16 * (2 * mipmapRow + 1)) + (2 * mipmapCol) + 1];
            }
        }

        return image;
    }

    explicit BitCountMipmapStack(const QuiccImage &quiccImage)
        : level3(computeLevel3(quiccImage)),
          level2(computeLevel2()),
          level1(computeLevel1()) {
        static_assert(spinImageWidthPixels == 64, "The index implementation of the library has been constructed for images of size 64x64");
    }

    void print() {
        std::cout << std::endl << "Level 1" << std::endl;
        for(unsigned int row = 0; row < 8; row++) {
            std::cout << "\t";
            for(unsigned int col = 0; col < 8; col++) {
                std::cout << (level1[row * 8 + col]) << (col < 7 ? ", " : "");
            }
            std::cout << std::endl;
        }

        std::cout << std::endl << "Level 2" << std::endl;
        for(unsigned int row = 0; row < 16; row++) {
            std::cout << "\t";
            for(unsigned int col = 0; col < 16; col++) {
                std::cout << (level2[row * 16 + col]) << (col < 15 ? ", " : "");
            }
            std::cout << std::endl;
        }

        std::cout << std::endl << "Level 3" << std::endl;
        for(unsigned int row = 0; row < 32; row++) {
            std::cout << "\t";
            for(unsigned int col = 0; col < 32; col++) {
                std::cout << int(level3[row * 32 + col]);
            }
            std::cout << std::endl;
        }
    }
};