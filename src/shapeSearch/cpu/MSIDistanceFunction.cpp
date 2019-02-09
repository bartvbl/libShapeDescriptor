#include <shapeSearch/libraryBuildSettings.h>
#include <assert.h>
#include "MSIDistanceFunction.h"

#ifdef _WIN32
#include <nmmintrin.h>
#endif

inline unsigned int popcount(unsigned long long bitstring) {
#ifdef _WIN32
	return _mm_popcnt_u64(bitstring);
#else
	return __builtin_popcountll(bitstring);
#endif
}

inline unsigned int compareImageRows(unsigned long long needleRow, unsigned long long hayStackRow) {

	unsigned long long equivalentBits = needleRow & hayStackRow;
	unsigned long long remainingBits = needleRow & (equivalentBits ^ 0xFFFFFFFFFFFFFFFF);
	unsigned int distance = 1;
	unsigned int distanceScore = 0;

	const unsigned int distanceLimit = 2;
	unsigned long long dilatedMask = needleRow;

	for(int i = 0; i < distanceLimit; i++) {
		dilatedMask = dilatedMask | ((needleRow >> distance) | (needleRow << distance));
	}

	while(remainingBits != 0 && distance < distanceLimit) {
		unsigned long long distancedPixels = ((hayStackRow >> distance) | (hayStackRow << distance)) & remainingBits;
		
		unsigned int distancePixelCount = popcount(distancedPixels);
		distanceScore += distancePixelCount * distance;
		distance += 1; 
		remainingBits = (distancedPixels ^ 0xFFFFFFFFFFFFFFFF) & remainingBits;
	}

	// This discounts pixels which attempt to abuse the system by spamming white pixels
	distanceScore += popcount(dilatedMask & (needleRow ^ hayStackRow));

	// Discount any non-accounted for pixels
	distanceScore += popcount(remainingBits) * distance;

	unsigned int score = distanceScore;//+ differenceInBitCount;

	return score;
}

unsigned int compareImages(const unsigned long long* needleImage, const unsigned long long* hayStackImage) {
	assert(spinImageWidthPixels == 8 * sizeof(unsigned long long));

	unsigned int totalScore = 0;
	for(int row = 0; row < spinImageWidthPixels; row++) {
		totalScore += compareImageRows(needleImage[row], hayStackImage[row]);
	}
	return totalScore;
}