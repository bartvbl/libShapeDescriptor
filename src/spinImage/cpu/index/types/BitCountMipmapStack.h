#pragma once

#include <spinImage/libraryBuildSettings.h>
#include <iostream>

//constexpr int uintsPerRow = spinImageWidthPixels / 32;

struct BitCountMipmapStack {


    //   level   mipmap size    pixel count   area per pixel   value range   space needed
    //   1       8x8 images     64            8x8 pixels       0-64          64 bytes, 8 (6) bits/pixel
    //   2       16x16 images   256           4x4 pixels       0-16          128 bytes, 4 bits/pixel
    //   3       32x32 images   1024          2x2 pixels       0-4           256 bytes, 2 bits/pixel
    //   -- 64x64: source --

    const std::array<unsigned char, 8*8> level1;
    const std::array<unsigned short, 16*16> level2;
    const std::array<unsigned short, 32*32> level3;

    BitCountMipmapStack(QuiccImage &quiccImage) {
        static_assert(spinImageWidthPixels == 64);

        computeMipmapLevel(level1, quiccImage, 8,  8, 0x01010101);
        computeMipmapLevel(level2, quiccImage, 4,  4, 0x11111111);


    }

    void print() {
        std::cout << "Level 0" << std::endl;
        for(int row = 0; row < 4; row++) {
            std::cout << "\t";
            for(int col = 0; col < 4; col++) {
                std::cout << ((level0[row] >> (24 - 8*col)) & 0xFF) << (col < 3 ? ", " : "");
            }
            std::cout << std::endl;
        }

        std::cout << std::endl << "Level 1" << std::endl;
        for(int row = 0; row < 8; row++) {
            std::cout << "\t";
            for(int col = 0; col < 8; col++) {
                std::cout << ((level1[uintsPerRow * row + col/4] >> (24 - 8*col)) & 0xFF) << (col < 7 ? ", " : "");
            }
            std::cout << std::endl;
        }

        std::cout << std::endl << "Level 2" << std::endl;
        for(int row = 0; row < 16; row++) {
            std::cout << "\t";
            for(int col = 0; col < 16; col++) {
                std::cout << ((level2[uintsPerRow * row + col/8] >> (28 - 4*col)) & 0xF) << (col < 15 ? ", " : "");
            }
            std::cout << std::endl;
        }

        std::cout << std::endl << "Level 3" << std::endl;
        for(int row = 0; row < 32; row++) {
            std::cout << "\t";
            for(int col = 0; col < 32; col++) {
                std::cout << ((level3[uintsPerRow * row + col/16] >> (30 - 2*col)) & 0x3) << (col < 31 ? ", " : "");
            }
            std::cout << std::endl;
        }
    }
};