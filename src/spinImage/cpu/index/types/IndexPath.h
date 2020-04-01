#pragma once

#include <array>
#include "BitCountMipmapStack.h"

#define INDEX_PATH_MAX_LENGTH 32

class IndexPath {
private:
    std::array<unsigned short, 32> pathDirections;
public:
    unsigned char length = 0;

private:
    std::array<unsigned short, 32> computeBitSequence(const BitCountMipmapStack &mipmapStack) {
        std::array<unsigned short, 32> columnSums;
        unsigned long cumulativeSum = mipmapStack.level1[0] + mipmapStack.level1[1] + mipmapStack.level1[2] + mipmapStack.level1[3];
        columnSums.at(0) = cumulativeSum;
        for(int column = 0; column < 32 - 1; column++) {
            for(int row = 0; row < 32; row++) {
                cumulativeSum -= mipmapStack.level6[32 * row + column];
            }
            columnSums.at(column + 1) = cumulativeSum;
        }
        return columnSums;
    }

    IndexPath(const std::array<unsigned short, 32> &existingPath, unsigned char existingPathLength) :
            pathDirections(existingPath), length(existingPathLength) {}
public:
    IndexPath(const BitCountMipmapStack &mipmapStack) :
            pathDirections(computeBitSequence(mipmapStack)) {}
    IndexPath() :
            pathDirections() {}

    static bool isBottomLevel(const IndexPath &path, unsigned int level) {
        // When remaining pixel count has reached 0, we've basically reached the bottom of the tree
        // At that point we can go over to a more 'linked list' style of storing images
        return (level >= INDEX_PATH_MAX_LENGTH) || (path.pathDirections[level] == 0);
    }

    unsigned long at(unsigned int level) {
        level = std::min<unsigned int>(INDEX_PATH_MAX_LENGTH - 1, level);
        return pathDirections.at(level);
    }

    unsigned int computeMinDistanceTo(const BitCountMipmapStack &referenceMipmapStack) {
        std::array<unsigned short, 32> referenceBitSequence = computeBitSequence(referenceMipmapStack);

        // If the path is empty, we can't say anything about the distance
        if(length == 0) {
            return 0;
        }

        unsigned int computedMinDistance = 0;

        // For the columns that are part of the path, we can compute the exact difference
        int column = 0;
        for(; column < int(length) - 1; column++) {
            int haystackColumnBitSum = (column < 31)
                    ? int(pathDirections[column]) - int(pathDirections[column + 1])
                    : int(pathDirections[column]);
            int needleColumnBitSum = (column < 31)
                    ? int(referenceBitSequence[column]) - int(referenceBitSequence[column + 1])
                    : int(referenceBitSequence[column]);

            // Lower than 0: Haystack must miss some needle set pixels
            // Greater than 0: Haystack must miss some needle unset pixels;
            int signedDelta = haystackColumnBitSum - needleColumnBitSum;
            computedMinDistance += std::abs(signedDelta);
        }

        // For the remainder, we look at the bit count difference for all columns
        if(length < 31) {
            int signedDelta = int(pathDirections[column]) - int(referenceBitSequence[column]);
            computedMinDistance += std::abs(signedDelta);
        }

        return computedMinDistance;
    }

    IndexPath append(unsigned long direction) {
        std::array<unsigned short, 32> newPath = pathDirections;
        if(length <= INDEX_PATH_MAX_LENGTH) {
            newPath.at(length) = direction;
        }
        return IndexPath(newPath, length + 1);
    }

    bool operator<(const IndexPath &otherPath) {
        if(length != otherPath.length) {
            return length < otherPath.length;
        }
        for(int i = 0; i < length; i++) {
            if(pathDirections[i] != otherPath.pathDirections[i]) {
                return pathDirections[i] < otherPath.pathDirections[i];
            }
        }
        // Both are equal, must be false per strict weak ordering requirement
        return false;
    }
};