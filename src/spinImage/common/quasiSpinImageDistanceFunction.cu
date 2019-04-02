#include "quasiSpinImageDistanceFunction.cuh"

const int indexBasedWarpCount = 16;

__inline__ __device__ int warpAllReduceSum(int val) {
    for (int mask = warpSize/2; mask > 0; mask /= 2)
        val += __shfl_xor_sync(0xFFFFFFFF, val, mask);
    return val;
}

__device__ __inline__ int compareQuasiSpinImagePairGPU(
        quasiSpinImagePixelType* needleImages,
        size_t needleImageIndex,
        quasiSpinImagePixelType* haystackImages,
        size_t haystackImageIndex) {
    int threadScore = 0;
    const int spinImageElementCount = spinImageWidthPixels * spinImageWidthPixels;
    const int laneIndex = threadIdx.x % 32;
    for(int row = laneIndex; row < spinImageWidthPixels; row++) {

        quasiSpinImagePixelType currentNeedlePixelValue =
                needleImages[needleImageIndex * spinImageElementCount + (row * spinImageWidthPixels + laneIndex)];
        quasiSpinImagePixelType currentHaystackPixelValue =
                haystackImages[haystackImageIndex * spinImageElementCount + (row * spinImageWidthPixels + laneIndex)];

        for(int col = laneIndex; col < spinImageWidthPixels - 32; col++) {
            quasiSpinImagePixelType nextNeedlePixelValue =
                    needleImages[needleImageIndex * spinImageElementCount + (row * spinImageWidthPixels + col)];
            quasiSpinImagePixelType nextHaystackPixelValue =
                    haystackImages[haystackImageIndex * spinImageElementCount + (row * spinImageWidthPixels + col)];

            quasiSpinImagePixelType nextRankNeedlePixelValue = __shfl_sync(0xFFFFFFFF,
                    // Input value
                                                                           (laneIndex == 0 ? nextNeedlePixelValue : currentNeedlePixelValue),
                    // Target thread
                                                                           (laneIndex == 31 ? 0 : threadIdx.x + 1));
            quasiSpinImagePixelType nextRankHaystackPixelValue = __shfl_sync(0xFFFFFFFF,
                    // Input value
                                                                             (laneIndex == 0 ? nextHaystackPixelValue : currentHaystackPixelValue),
                    // Target thread
                                                                             (laneIndex == 31 ? 0 : threadIdx.x + 1));

            quasiSpinImagePixelType needleDelta = nextRankNeedlePixelValue - currentNeedlePixelValue;
            quasiSpinImagePixelType haystackDelta = nextRankHaystackPixelValue - currentHaystackPixelValue;

                threadScore += (needleDelta - haystackDelta) * (needleDelta - haystackDelta);

            currentNeedlePixelValue = nextNeedlePixelValue;
            currentHaystackPixelValue = nextHaystackPixelValue;
        }

        quasiSpinImagePixelType nextRankNeedlePixelValue = __shfl_sync(0xFFFFFFFF, currentNeedlePixelValue, laneIndex + 1);
        quasiSpinImagePixelType nextRankHaystackPixelValue = __shfl_sync(0xFFFFFFFF, currentHaystackPixelValue, laneIndex + 1);

        quasiSpinImagePixelType needleDelta = nextRankNeedlePixelValue - currentNeedlePixelValue;
        quasiSpinImagePixelType haystackDelta = nextRankHaystackPixelValue - currentHaystackPixelValue;

        threadScore += (needleDelta - haystackDelta) * (needleDelta - haystackDelta);
    }

    int imageScore = warpAllReduceSum(threadScore);

    return imageScore;
}