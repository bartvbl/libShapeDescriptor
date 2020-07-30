#pragma once

#include <utility>
#include <spinImage/libraryBuildSettings.h>
#include <algorithm>
#include <bitset>

namespace SpinImage {
    namespace utilities {
        struct HammingWeights {
            float missingSetBitPenalty = 0;
            float missingUnsetBitPenalty = 0;
        };

        #ifdef __CUDACC__
            __device__
        #endif
        inline HammingWeights computeWeightedHammingWeights(
                unsigned int setBitCount, unsigned int totalBitsInBitString) {
            unsigned int queryImageUnsetBitCount = totalBitsInBitString - setBitCount;

            // If any count is 0, bump it up to 1

            #ifdef __CUDACC__
                setBitCount = max(setBitCount, 1);
                queryImageUnsetBitCount = max(queryImageUnsetBitCount, 1);
            #else
                setBitCount = std::max<unsigned int>(setBitCount, 1);
                queryImageUnsetBitCount = std::max<unsigned int>(queryImageUnsetBitCount, 1);
            #endif

            // The fewer bits exist of a specific pixel type, the greater the penalty for not containing it
            float missedSetBitPenalty = float(totalBitsInBitString) / float(setBitCount);
            float missedUnsetBitPenalty = float(totalBitsInBitString) / float(queryImageUnsetBitCount);

            return {missedSetBitPenalty, missedUnsetBitPenalty};
        }

        #ifdef __CUDACC__
                __device__
        #endif
        inline float computeChunkWeightedHammingDistance(HammingWeights hammingWeights, const unsigned int needle, const unsigned int haystack) {
            #ifdef __CUDACC__
                unsigned int missingSetPixelCount = __popc((needle ^ haystack) & needle);
                unsigned int missingUnsetPixelCount = __popc((~needle ^ ~haystack) & ~needle);
            #else
                unsigned int missingSetPixelCount = std::bitset<32>((needle ^ haystack) & needle).count();
                unsigned int missingUnsetPixelCount = std::bitset<32>((~needle ^ ~haystack) & ~needle).count();
            #endif

            return float(missingSetPixelCount) * hammingWeights.missingSetBitPenalty +
                   float(missingUnsetPixelCount) * hammingWeights.missingUnsetBitPenalty;
        }

        #ifdef __CUDACC__
                __device__
        #endif
        inline float computeWeightedHammingDistance(HammingWeights hammingWeights, const unsigned int* needle, const unsigned int* haystack, unsigned int length) {
            float distanceScore = 0;

            for(unsigned int i = 0; i < length; i++) {
                distanceScore += computeChunkWeightedHammingDistance(hammingWeights, needle[i], haystack[i]);
            }

            return distanceScore;
        }
    }
}