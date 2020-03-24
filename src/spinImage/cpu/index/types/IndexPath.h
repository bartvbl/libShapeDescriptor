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

    float computeMinDistanceTo(const BitCountMipmapStack &referenceMipmapStack) {
        std::array<unsigned short, 32> referenceBitSequence = computeBitSequence(referenceMipmapStack);

        // If the path is empty, we can't say anything about the distance
        if(length == 0) {
            return 0;
        }

        const unsigned int bitsPerImage = spinImageWidthPixels * spinImageWidthPixels;
        unsigned int queryImageSetBitCount = referenceMipmapStack.level1[0] + referenceMipmapStack.level1[1]
                                             + referenceMipmapStack.level1[2] + referenceMipmapStack.level1[3];
        unsigned int queryImageUnsetBitCount = bitsPerImage - queryImageSetBitCount;

        // If any count is 0, bump it up to 1
        queryImageSetBitCount = std::max<unsigned int>(queryImageSetBitCount, 1);
        queryImageUnsetBitCount = std::max<unsigned int>(queryImageUnsetBitCount, 1);

        // The fewer bits exist of a specific pixel type, the greater the penalty for not containing it
        float missedSetBitPenalty = float(bitsPerImage) / float(queryImageSetBitCount);
        float missedUnsetBitPenalty = float(bitsPerImage) / float(queryImageUnsetBitCount);

        // For the columns that are part of the path, we can compute the exact difference
        float computedMinDistance = 0;
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

            computedMinDistance += signedDelta < 0
                    ? float(std::abs(signedDelta)) * missedSetBitPenalty
                    : float(std::abs(signedDelta)) * missedUnsetBitPenalty;
        }

        // For the remainder, we look at the bit count difference for all columns
        if(length < 31) {
            int signedDelta = int(pathDirections[column]) - int(referenceBitSequence[column]);

            computedMinDistance += signedDelta < 0
                   ? float(std::abs(signedDelta)) * missedSetBitPenalty
                   : float(std::abs(signedDelta)) * missedUnsetBitPenalty;
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