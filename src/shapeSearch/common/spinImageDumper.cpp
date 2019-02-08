#include "spinImageDumper.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <bitset>


#include <vector>

template<typename spinPixelType>
void performSpinDump(array<spinPixelType> descriptors, OutputImageSettings imageSettings, unsigned int imagesPerRow) {
	size_t rowCount = (descriptors.length / imagesPerRow) + 1;
	std::cout << "Dumping " << rowCount << " rows containing " << descriptors.length << " images." << std::endl;

	unsigned int width = imagesPerRow * (spinImageWidthPixels + 1);
	size_t height = rowCount * (spinImageWidthPixels + 1);

	spinPixelType max = 0;
	size_t nonzeroPixelCount = 0;

	for (size_t image = 0; image < descriptors.length; image++)
	{
		for(size_t x = 0; x < spinImageWidthPixels; x++)
		{
			for(size_t y = 0; y < spinImageWidthPixels; y++)
			{
				size_t pixel_index = size_t(spinImageWidthPixels) * size_t(spinImageWidthPixels) * image + size_t(spinImageWidthPixels) * y + x;
				spinPixelType pixel_value = descriptors.content[pixel_index];
				if(pixel_value != 0) {
					nonzeroPixelCount++;
				}
				if(!std::isnan(pixel_value)) {
					max = std::max(pixel_value, max);
				}
			}
		}
	}
	std::cout << "Image dumper: max is " << max << std::endl;
	std::cout << "Image dumper: nonzero pixel count is: " << nonzeroPixelCount << std::endl;

	std::vector<unsigned char> imageData;
	size_t pixelCount = width * height * 4;
	imageData.resize(pixelCount);
	size_t imageIndex = 0;

	// Initialise the images to default values.
	// Unused images will also be marked with a red cross
	for(size_t row = 0; row < rowCount; row++)
	{
		for(size_t col = 0; col < imagesPerRow; col++)
		{
			for (size_t x = 0; x < spinImageWidthPixels; x++)
			{
				for (size_t y = 0; y < spinImageWidthPixels; y++)
				{
					size_t pixelX = col * (spinImageWidthPixels + 1) + x;
					size_t pixelY = row * (spinImageWidthPixels + 1) + (spinImageWidthPixels - 1 - y);
					size_t pixelBaseIndex = 4 * (pixelX + width * pixelY);

					imageData[pixelBaseIndex + 0] = 255;
					imageData[pixelBaseIndex + 1] = 255;
					imageData[pixelBaseIndex + 2] = 255;
					imageData[pixelBaseIndex + 3] = 255;

					if (((x == y) || (x == spinImageWidthPixels - 1 - y)) && x < spinImageWidthPixels && y < spinImageWidthPixels)
					{
						imageData[pixelBaseIndex + 1] = 0;
						imageData[pixelBaseIndex + 2] = 0;
					}
				}
			}
			imageIndex++;
		}
	}

	imageIndex = 0;

	for(size_t row = 0; row < rowCount; row++)
	{
		for(size_t col = 0; col < imagesPerRow; col++)
		{
			if (imageIndex >= descriptors.length)
			{
				break; //stop once we have reached the final spin image
			}

			for (size_t x = 0; x < spinImageWidthPixels; x++)
			{
				for (size_t y = 0; y < spinImageWidthPixels; y++)
				{
					size_t pixel_index = size_t(spinImageWidthPixels) * size_t(spinImageWidthPixels) * imageIndex + size_t(spinImageWidthPixels) * y + x;
					spinPixelType pixelValue = descriptors.content[pixel_index];
					float normalised;
					if (imageSettings.enableLogImage) {
						normalised = (std::log(std::max(0.0f, float(pixelValue))) / float(std::log(max))) * 255.0f;
					} else
					{
						normalised = (std::max(0.0f, float(pixelValue)) / float(max)) * 255.0f;
					}
					unsigned char pixelByte = (unsigned char)unsigned(std::min(std::max(normalised, 0.0f), 255.0f));

					size_t pixelX = col * (spinImageWidthPixels + 1) + x;
					size_t pixelY = row * (spinImageWidthPixels + 1) + (spinImageWidthPixels - 1 - y); // Flip image because image coordinates

					size_t pixelBaseIndex = 4 * (pixelX + width * pixelY);

					if(!std::isnan(pixelValue)) {
						imageData[pixelBaseIndex + 0] = pixelByte;
						imageData[pixelBaseIndex + 1] = pixelByte;
						imageData[pixelBaseIndex + 2] = pixelByte;
						imageData[pixelBaseIndex + 3] = 255;
					} else {
						imageData[pixelBaseIndex + 0] = 255;
						imageData[pixelBaseIndex + 1] = 0;
						imageData[pixelBaseIndex + 2] = 0;
						imageData[pixelBaseIndex + 3] = 255;
					}
				}
			}

			imageIndex++;
		}
	}

	std::cout << "Writing image file.. " << imageSettings.imageDestinationFile << std::endl;

	unsigned error = lodepng::encode(imageSettings.imageDestinationFile, imageData, width, height);

	if(error)
	{
		std::cout << "encoder error " << error << ": " << lodepng_error_text(error) << std::endl;
	}
}

void dumpImages(VertexDescriptors descriptors, OutputImageSettings imageSettings, unsigned int imagesPerRow)
{
	if(descriptors.isClassic) {
		performSpinDump<classicSpinImagePixelType> (descriptors.classicDescriptorArray, imageSettings);
	} else if(descriptors.isNew) {
		performSpinDump<newSpinImagePixelType> (descriptors.newDescriptorArray, imageSettings);
	}

}

void dumpCompressedImages(array<unsigned int> compressedDescriptors, OutputImageSettings imageSettings, unsigned int imagesPerRow) {
	array<unsigned int> decompressedDesciptors;
	size_t imageTotalPixelCount = compressedDescriptors.length * spinImageWidthPixels * spinImageWidthPixels;
	
	
	

	decompressedDesciptors.content = new unsigned int[imageTotalPixelCount];
	decompressedDesciptors.length = compressedDescriptors.length;



	for(int compressedEntry = 0; compressedEntry < imageTotalPixelCount / 32; compressedEntry++) {
		unsigned int entry = compressedDescriptors.content[compressedEntry];
		std::bitset<32> entryBits(entry);
		//std::cout << entry << ", ";
		for(int bitInEntry = 0; bitInEntry < 32; bitInEntry++) {
			int pixelIndex = 32 * compressedEntry + bitInEntry;

		    /*unsigned int outputIndex = compressedEntry * 32 + bitInEntry;


			const unsigned int warpSize = 32;
			const unsigned int cacheLinesPerRow = spinImageWidthPixels / warpSize;
			const unsigned int cacheLinesPerImage = cacheLinesPerRow * spinImageWidthPixels;

			size_t offsetWithinImage = compressedEntry - (imageIndex * elementsPerCompressedImage);
			size_t imageRow = offsetWithinImage / cacheLinesPerRow;
			size_t cacheLineInRow = offsetWithinImage - (imageRow * cacheLinesPerRow);
			size_t x = (cacheLineInRow * 32) + bitInEntry;
			size_t y = imageRow;*/

			//if(imageIndex == 100 && imageRow == 20) {
			//	std::cout << "Row: " << imageRow << ", Column: " << x << " -> " << std::hex << entry << std::dec << std::endl;
			//}

			
			//	std::cout << compressedEntry << " / " << elementsPerCompressedImage << " -> (" << imageIndex << ", " << x << ", " << y << ") " << std::endl;

			decompressedDesciptors.content[pixelIndex] = (entryBits[31 - bitInEntry] * 255);
		}
	}

	imageSettings.imageDestinationFile = imageSettings.compressedDestinationFile;

	performSpinDump<unsigned int>(decompressedDesciptors, imageSettings);
}

void dumpRawCompressedImages(array<unsigned int> compressedDescriptors, std::string destination, unsigned int imagesPerRow) {
	std::cout << "Dumping raw compressed images to: " << destination << std::endl;

	unsigned int itemCount = unsigned(compressedDescriptors.length);
	//size_t bufferSize = sizeof(unsigned int) * itemCount;
	size_t integerCount = ((spinImageWidthPixels * spinImageWidthPixels) / (sizeof(unsigned int) * 8)) * itemCount;
	size_t arrayLength = integerCount * sizeof(unsigned int);
	
	unsigned int header = 0x44534352;
	unsigned int spinImageWidth = spinImageWidthPixels;

	std::ofstream outFile;
	outFile.open(destination, std::ofstream::out | std::ofstream::binary);

	if(outFile.bad() || !outFile.is_open())
	{
		std::cout << "Could not open file! " << std::endl;
	}

	outFile.write((char*) &header, sizeof(unsigned int));
	outFile.write((char*) &itemCount, sizeof(unsigned int));
	outFile.write((char*) &spinImageWidth, sizeof(unsigned int));
	outFile.write((char*) compressedDescriptors.content, arrayLength);
	outFile.flush();
	outFile.close();

	if(outFile.bad())
	{
		std::cout << "Could not close file! " << std::endl;
	}

	std::cout << "Dump complete." << std::endl;
}