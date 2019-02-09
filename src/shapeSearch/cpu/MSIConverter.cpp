#include <bitset>
#include <shapeSearch/libraryBuildSettings.h>
#include "MSIConverter.h"

void compressMSI(array<unsigned int> msiImage, unsigned long long* compressedImage) {
	for(unsigned int row = 0; row < spinImageWidthPixels; row++) {
		unsigned long long pixelRow = 0;
		std::bitset<spinImageWidthPixels> pixelRowBits(pixelRow);
		for(unsigned int col = 0; col < spinImageWidthPixels; col++) {
			// bitset uses index 0 as the least significant bit. We invert that here.
		    pixelRowBits[spinImageWidthPixels - 1 - col] = msiImage.content[spinImageWidthPixels * row + col] == 1 ? 1 : 0;
		}
		compressedImage[row] = pixelRowBits.to_ullong();
	}
}