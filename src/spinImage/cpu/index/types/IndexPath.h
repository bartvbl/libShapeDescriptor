#pragma once

#include <array>
#include "BitCountMipmapStack.h"

#define INDEX_PATH_MAX_LENGTH 32

class IndexPath {
private:
    std::vector<unsigned long> pathDirections;
public:
    size_t length() {
        return pathDirections.size();
    }

private:
    std::vector<unsigned long> computeBitSequence(const BitCountMipmapStack &mipmapStack) {
        std::vector<unsigned long> columnSums;
        unsigned long cumulativeSum = mipmapStack.level1[0] + mipmapStack.level1[1] + mipmapStack.level1[2] + mipmapStack.level1[3];
        columnSums.resize(32);
        for(int column = 0; column < 32; column++) {
            for(int row = 0; row < 32; row++) {
                cumulativeSum -= mipmapStack.level6[32 * row + column];
            }
            columnSums.at(column) = cumulativeSum;
        }
        return columnSums;
    }

    IndexPath(const std::vector<unsigned long> &existingPath) :
            pathDirections(existingPath) {}
public:
    IndexPath(const BitCountMipmapStack &mipmapStack) :
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

    unsigned short computeDeltaAt(std::vector<unsigned long> &reference, unsigned int index) {
        return std::abs((signed long) (reference[index]) - (signed long) (pathDirections[index]));
    }

    unsigned short computeMinDistanceTo(const BitCountMipmapStack &mipmapStack) {
        std::vector<unsigned long> referenceBitSequence = computeBitSequence(mipmapStack);

        unsigned short computedMinDistance = 0;
        for(unsigned int i = 0; i < length(); i++) {
            computedMinDistance += computeDeltaAt(referenceBitSequence, i);
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