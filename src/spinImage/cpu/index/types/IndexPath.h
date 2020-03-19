#pragma once

#include <array>
#include "BitCountMipmapStack.h"

#define INDEX_PATH_MAX_LENGTH 8
#define INDEX_PATH_INITIAL_MAX ((spinImageWidthPixels * spinImageWidthPixels) / 8)

class IndexPath {
private:
    std::vector<unsigned long> pathDirections;
public:
    size_t length() {
        return pathDirections.size();
    }

private:
    template<unsigned int width, unsigned int height> unsigned long computeSingleBitSequence(
            const std::array<unsigned short, width*height> &image,
            std::array<unsigned short, width*height>* mins,
            std::array<unsigned short, width*height>* maxes) {
        unsigned long bitSequence = 0;

        for(unsigned int i = 0; i < width * height; i++) {
            unsigned short pivot = (maxes->at(i) - mins->at(i)) / 2;
            bool directionBit = image[i] >= pivot;
            bitSequence = bitSequence | (((unsigned int) directionBit) << (width * height - 1 - i));
            if(directionBit) {
                mins->at(i) = pivot;
            } else {
                maxes->at(i) = pivot;
            }
        }

        return bitSequence;
    }

    void computeBitSequence(
            BitCountMipmapStack mipmapStack,
            std::array<unsigned short, 8>* mins,
            std::array<unsigned short, 8>* maxes,
            std::vector<unsigned long>* bitSequence) {
        *mins = {0, 0, 0, 0, 0, 0, 0, 0};
        *maxes = {INDEX_PATH_INITIAL_MAX, INDEX_PATH_INITIAL_MAX, INDEX_PATH_INITIAL_MAX, INDEX_PATH_INITIAL_MAX,
                 INDEX_PATH_INITIAL_MAX, INDEX_PATH_INITIAL_MAX, INDEX_PATH_INITIAL_MAX, INDEX_PATH_INITIAL_MAX};

        for(int i = 0; i < length(); i++) {
            bitSequence->push_back(computeSingleBitSequence<2, 4>(mipmapStack.level2, mins, maxes));
        }
    }

    std::vector<unsigned long> computeBitSequence(
            BitCountMipmapStack mipmapStack) {
        std::array<unsigned short, 8> mins;
        std::array<unsigned short, 8> maxes;
        std::vector<unsigned long> bitSequence;
        computeBitSequence(mipmapStack, &mins, &maxes, &bitSequence);
        return bitSequence;
    }

    IndexPath(const std::vector<unsigned long> &existingPath) :
            pathDirections(existingPath) {}
public:
    IndexPath(BitCountMipmapStack mipmapStack) :
            pathDirections(computeBitSequence(mipmapStack)) {}
    IndexPath() :
            pathDirections() {}

    bool isBottomLevel(unsigned int level) {
        return level >= INDEX_PATH_MAX_LENGTH;
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
        computeBitSequence(mipmapStack, &mins, &maxes, &bitSequence);


        for(int i = 0; i < 8; i++) {
            int deltas = int(mins[i]) + int(mipmapStack.level2[i]) - int(1024);

            // For index i, the number of bits set for all images whose paths start with this one
            // lie between mins[i] and maxes[i].
            if(mins[i] > mipmapStack.level2[i]) {
                computedMinDistance += mins[i] - mipmapStack.level2[i];
            } else if(maxes[i] < mipmapStack.level2[i]) {
                computedMinDistance += mipmapStack.level2[i] - maxes[i];
            }/* else if(mins[i] == 0 ^ mipmapStack.level2[i] == 0) {
                computedMinDistance += mins[i] + mipmapStack.level2[i];
            }*/
            else if (deltas > 0) {
                computedMinDistance += deltas;
            }
        }

        return computedMinDistance;
    }

    IndexPath append(unsigned long direction) {
        std::vector<unsigned long> newPath = pathDirections;
        if(length() <= INDEX_PATH_MAX_LENGTH) {
            newPath.push_back(direction);
        }
        return IndexPath(newPath);
    }
};