#pragma once

#include <utility>
#include <shapeDescriptor/shapeDescriptor.h>
#include <algorithm>
#include <bitset>

#ifndef __CUDACC__
    unsigned int __popc(unsigned int x);
#endif

namespace ShapeDescriptor {
    namespace utilities {
        struct HammingWeights {
            float missingSetBitPenalty = 0;
            float missingUnsetBitPenalty = 0;
        };

        #ifdef __CUDACC__
            __host__
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
        __host__
        #endif
        inline HammingWeights computeWeightedHammingWeights(const ShapeDescriptor::QUICCIDescriptor &descriptor) {
            unsigned int setBitCount = 0;

            for(unsigned int i = 0; i < (spinImageWidthPixels * spinImageWidthPixels) / (8 * sizeof(uint32_t)); i++) {
                setBitCount += std::bitset<32>(descriptor.contents[i]).count();
            }

            return computeWeightedHammingWeights(setBitCount, spinImageWidthPixels * spinImageWidthPixels);
        }

#ifdef __CUDACC__
        __device__
#endif
        inline HammingWeights computeWeightedHammingWeightsGPU(const ShapeDescriptor::QUICCIDescriptor &descriptor) {
            unsigned int setBitCount = 0;

            for(unsigned int i = 0; i < (spinImageWidthPixels * spinImageWidthPixels) / (8 * sizeof(uint32_t)); i++) {
                setBitCount += __popc(descriptor.contents[i]);
            }

            return computeWeightedHammingWeights(setBitCount, spinImageWidthPixels * spinImageWidthPixels);
        }

        #ifdef __CUDACC__
                __device__
        #endif
        inline float computeChunkWeightedHammingDistance(const HammingWeights hammingWeights,
                                                         const unsigned int needle,
                                                         const unsigned int haystack) {
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
        inline float computeWeightedHammingDistance(const HammingWeights hammingWeights,
                                                    const unsigned int* needle,
                                                    const unsigned int* haystack,
                                                    const unsigned int imageWidthBits,
                                                    const unsigned int imageHeightBits) {
            float distanceScore = 0;

            const unsigned int chunksPerRow = imageWidthBits / (8 * sizeof(unsigned int));

            for(unsigned int row = 0; row < imageHeightBits; row++) {
                for(unsigned int col = 0; col < chunksPerRow; col++) {
                    distanceScore += computeChunkWeightedHammingDistance(hammingWeights,
                                                                         needle[row * chunksPerRow + col],
                                                                         haystack[row * chunksPerRow + col]);
                }
            }

            return distanceScore;
        }
    }
}