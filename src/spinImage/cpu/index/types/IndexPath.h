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
        columnSums.at(0) = cumulativeSum;
        for(int column = 0; column < 32 - 1; column++) {
            for(int row = 0; row < 32; row++) {
                cumulativeSum -= mipmapStack.level6[32 * row + column];
            }
            columnSums.at(column + 1) = cumulativeSum;
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

    unsigned short computeDeltaAt(std::vector<unsigned long> &referenceSums, unsigned int index) {
        int pathDeltaValue = (index < 31) ? int(pathDirections[index]) - int(pathDirections[index + 1]) : int(pathDirections[index]);
        int referenceDeltaValue = (index < 31) ? int(referenceSums[index]) - int(referenceSums[index + 1]) : int(referenceSums[index]);
        return std::abs(pathDeltaValue - referenceDeltaValue);
    }

    unsigned short computeMinDistanceTo(const BitCountMipmapStack &referenceMipmapStack) {
        std::vector<unsigned long> referenceBitSequence = computeBitSequence(referenceMipmapStack);

        // If the path is empty, we can't say anything about the distance
        if(length() == 0) {
            return 0;
        }

        // For the columns that are part of the path, we can compute the exact difference
        unsigned short computedMinDistance = 0;
        int column = 0;
        for(; column < int(length()) - 1; column++) {
            computedMinDistance += computeDeltaAt(referenceBitSequence, column);
        }

        // For the remainder, we look at the bit count difference for all columns
        if(length() < 31) {
            computedMinDistance += std::abs(int(pathDirections[column]) - int(referenceBitSequence[column]));
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