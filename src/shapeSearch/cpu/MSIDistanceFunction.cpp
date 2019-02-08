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

	//const unsigned char requiredOnes[] = {0b00000100, 0b00000010, 0b00001000, 0b00000100};
	
	unsigned long long remainingBits = needleRow & (equivalentBits ^ 0xFFFFFFFFFFFFFFFF);
	unsigned int distance = 1;
	unsigned int distanceScore = 0;
	//std::cout << "First: " << std::hex << equivalentBits << std::dec << " -> " << std::hex << (equivalentBits ^ 1) << std::dec << std::endl;
	//std::cout << distance  << " - " << std::hex << needleRow << std::dec << " vs " << std::hex << hayStackRow << std::dec << ": " << std::hex << remainingBits << std::dec << std::endl;

	//const unsigned int distanceLimit = 64;

	

	const unsigned int distanceLimit = 2;
	unsigned long long dilatedMask = needleRow;
	//const unsigned int guassianWeights[] = {35, 15, 1};

	for(int i = 0; i < distanceLimit; i++) {
		dilatedMask = dilatedMask | ((needleRow >> distance) | (needleRow << distance));
	}
		

	while(remainingBits != 0 && distance < distanceLimit) {
		unsigned long long distancedPixels = ((hayStackRow >> distance) | (hayStackRow << distance)) & remainingBits;
		
		unsigned int distancePixelCount = popcount(distancedPixels);
		distanceScore += distancePixelCount * distance;
		//distanceScore += distancePixelCount * guassianWeights[distance];
		distance += 1; 
		remainingBits = (distancedPixels ^ 0xFFFFFFFFFFFFFFFF) & remainingBits;
			//std::cout << distance << ": " << std::hex << remainingBits << std::dec << " - " << distancePixelCount << std::endl;
	}

	/*std::cout 
	<< std::hex << dilatedMask << std::dec << " & (" 
	<< std::hex << needleRow << std::dec << " ^ " 
	<< std::hex << hayStackRow << std::dec << ") = " 
	<< std::hex << (dilatedMask & (needleRow ^ hayStackRow)) << std::dec 
	<< " -> " << std::hex << (needleRow ^ hayStackRow) << std::hex
	<< " -> " << __builtin_popcountll((dilatedMask & (needleRow ^ hayStackRow))) << std::endl;*/
	
	// This discounts pixels which attempt to abuse the system by spamming white pixels
	distanceScore += popcount(dilatedMask & (needleRow ^ hayStackRow));

	// Discount any non-accounted for pixels
	distanceScore += popcount(remainingBits) * distance;
	
	//unsigned long long remainingBits = needleRow & !equivalentBits & !bitsOnePixelAway;

	//unsigned int needleBitCount = __builtin_popcountll(needleRow);
	//unsigned int haystackBitCount = __builtin_popcountll(hayStackRow);

	
	//unsigned int remainingBitCount = __builtin_popcountll(remainingBits);

	//unsigned int differenceInBitCount = std::abs(int(needleBitCount) - int(haystackBitCount));



	unsigned int score = distanceScore;//+ differenceInBitCount;

	//std::cout << "Score = " << score << " -> " << distanceScore << " + " << differenceInBitCount << std::endl;

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