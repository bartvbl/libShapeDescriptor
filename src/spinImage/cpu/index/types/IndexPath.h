#pragma once

#include <array>
#include "BitCountMipmapStack.h"

#define INDEX_PATH_MAX_LENGTH 8
#define INDEX_PATH_INITIAL_MAX ((spinImageWidthPixels * spinImageWidthPixels) / 8)

class IndexPath {
private:
    std::vector<unsigned long> pathDirections;
public:
    unsigned int length;
private:

    template<unsigned int width, unsigned int height> unsigned long computeSingleBitSequence(
            const std::array<unsigned short, width*height> &image, 
            std::array<unsigned short, width*height> &mins, 
            std::array<unsigned short, width*height> &maxes) {
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

    void computeBitSequence(
            BitCountMipmapStack mipmapStack,
            std::array<unsigned short, 8> &mins,
            std::array<unsigned short, 8> &maxes,
            std::vector<unsigned long> &bitSequence) {
        mins = {0, 0, 0, 0, 0, 0, 0, 0};
        maxes = {INDEX_PATH_INITIAL_MAX, INDEX_PATH_INITIAL_MAX, INDEX_PATH_INITIAL_MAX, INDEX_PATH_INITIAL_MAX,
                 INDEX_PATH_INITIAL_MAX, INDEX_PATH_INITIAL_MAX, INDEX_PATH_INITIAL_MAX, INDEX_PATH_INITIAL_MAX};

        for(int i = 0; i < length; i++) {
            bitSequence.push_back(computeSingleBitSequence<2, 4>(mipmapStack.level2, mins, maxes));
        }
    }

    std::vector<unsigned long> computeBitSequence(
            BitCountMipmapStack mipmapStack) {
        std::array<unsigned short, 8> mins;
        std::array<unsigned short, 8> maxes;
        std::vector<unsigned long> bitSequence;
        computeBitSequence(mipmapStack, mins, maxes, bitSequence);
        return bitSequence;
    }

    IndexPath(const std::vector<unsigned long> &existingPath) :
            pathDirections(existingPath),
            length(existingPath.size()) {}
public:
    IndexPath(BitCountMipmapStack mipmapStack) :
            pathDirections(computeBitSequence(mipmapStack)),
            length(INDEX_PATH_MAX_LENGTH) {}
    IndexPath() :
            pathDirections({0, 0, 0, 0, 0, 0, 0, 0}),
            length(0) {}

    bool isBottomLevel() {
        return length >= INDEX_PATH_MAX_LENGTH;
    }

    unsigned long at(unsigned int level) {
        level = std::min<unsigned int>(INDEX_PATH_MAX_LENGTH - 1, level);
        return pathDirections.at(level);
    }

    unsigned short computeMinDistanceTo(const BitCountMipmapStack &mipmapStack) {
        unsigned short computedMinDistance = 0;

        std::array<unsigned short, 8> mins;
        std::array<unsigned short, 8> maxes;
        std::vector<unsigned long> bitSequence;
        computeBitSequence(mipmapStack, mins, maxes, bitSequence);

        for(int i = 0; i < 8; i++) {
            // For index i, the number of bits set for all images whose paths start with this one
            // lie between mins[i] and maxes[i].
            if((mins[i] == 0) ^ (mipmapStack.level2[i] == 0)) {
                computedMinDistance += mins[i] + mipmapStack.level2[i];
            }
        }

        return computedMinDistance;
    }

    IndexPath append(unsigned long direction) {
        std::vector<unsigned long> newPath = pathDirections;
        if(length <= INDEX_PATH_MAX_LENGTH) {
            newPath.push_back(direction);
        }
        return IndexPath(newPath);
    }
private:
    std::string byteToHex(unsigned char byte) {
        std::string byteString;
        const std::string characterMap = "0123456789abcdef";
        byteString += characterMap.at((byte >> 4U) & 0x0FU);
        byteString += characterMap.at((byte & 0x0FU));
        return byteString;
    }
public:
    std::string to_string() {
        std::stringstream pathBuilder;
        pathBuilder << std::hex;
        // childNodeID + byteToHex(childIndex) + "/";
        // pathBuilder << (outgoingEdgeIndex < 16 ? "0" : "") << int(outgoingEdgeIndex) << "/";

        // Default append
        // pathBuilder << (outgoingEdgeIndex < 16 ? "0" : "") << int(outgoingEdgeIndex) << "/";
        // currentNodeID = pathBuilder.str();
    }
};