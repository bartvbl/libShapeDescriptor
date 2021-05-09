#include "descriptorImages.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <bitset>
#include <shapeDescriptor/utilities/free/array.h>


#include <vector>
#include <lodepng.h>

template<typename spinPixelType>
unsigned char computeImageByte(bool logarithmicImage, spinPixelType max, spinPixelType pixelValue) {
    unsigned char pixelByte;
    float normalised;
    if (logarithmicImage && max != 1.0f) {
        normalised = (std::log(std::max(0.0f, float(pixelValue))) / std::log(float(max))) * 255.0f;
    } else {
        normalised = (std::max(0.0f, float(pixelValue)) / float(max)) * 255.0f;
    }
    pixelByte = (unsigned char) unsigned(std::min(std::max(normalised, 0.0f), 255.0f));
    return pixelByte;
}

template<typename spinPixelType, typename descriptorType>
void performSpinDump(ShapeDescriptor::cpu::array<descriptorType> redChannelDescriptors,
                     ShapeDescriptor::cpu::array<descriptorType> greenChannelDescriptors,
                     ShapeDescriptor::cpu::array<descriptorType> blueChannelDescriptors,
                     const std::experimental::filesystem::path &imageDestinationFile,
                     bool logarithmicImage,
                     unsigned int imagesPerRow) {
	size_t rowCount = (redChannelDescriptors.length / imagesPerRow) + ((redChannelDescriptors.length % imagesPerRow == 0) ? 0 : 1);
	std::cout << "Dumping " << rowCount << " rows containing " << redChannelDescriptors.length << " images." << std::endl;

	unsigned int width = imagesPerRow * (spinImageWidthPixels + 1);
	size_t height = rowCount * (spinImageWidthPixels + 1);

	spinPixelType redMax = 0;
    spinPixelType greenMax = 0;
    spinPixelType blueMax = 0;
	size_t nonzeroPixelCount = 0;

	for (size_t image = 0; image < redChannelDescriptors.length; image++)
	{
		for(size_t x = 0; x < spinImageWidthPixels; x++)
		{
			for(size_t y = 0; y < spinImageWidthPixels; y++)
			{
				size_t pixel_index = size_t(spinImageWidthPixels) * y + x;
				spinPixelType redChannelPixelValue = redChannelDescriptors.content[image].contents[pixel_index];
                spinPixelType greenChannelPixelValue = greenChannelDescriptors.content[image].contents[pixel_index];
                spinPixelType blueChannelPixelValue = blueChannelDescriptors.content[image].contents[pixel_index];
				if(redChannelPixelValue != 0 || greenChannelPixelValue != 0 || blueChannelPixelValue != 0) {
					nonzeroPixelCount++;
				}
				if(!std::isnan((float) redChannelPixelValue) && redChannelPixelValue != UINT32_MAX) {
					redMax = std::max(redChannelPixelValue, redMax);
				}
                if(!std::isnan((float) greenChannelPixelValue) && greenChannelPixelValue != UINT32_MAX) {
                    greenMax = std::max(greenChannelPixelValue, greenMax);
                }
				if(!std::isnan((float) blueChannelPixelValue) && blueChannelPixelValue != UINT32_MAX) {
                    blueMax = std::max(blueChannelPixelValue, blueMax);
                }

			}
		}
	}
	std::cout << "Image dumper: max is " << std::max(redMax, std::max(greenMax, blueMax)) << std::endl;
	std::cout << "Image dumper: nonzero pixel count is: " << nonzeroPixelCount << std::endl;

	if(redMax == 1 || greenMax == 1 || blueMax == 1) {
		std::cout << "WARNING: ignoring logarithmic image parameter, as maximum pixel value is 1 (would cause 0 division)." << std::endl;
		logarithmicImage = false;
	}

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

					// These create a red X in the middle by turning off the green and blue channels
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
			if (imageIndex >= redChannelDescriptors.length)
			{
				break; //stop once we have reached the final spin image
			}

			// Hacky way to mark image as empty. Used to display multiple sets of images.
			unsigned int topLeftPixel = redChannelDescriptors.content[imageIndex].contents[0];
			if(topLeftPixel != UINT32_MAX) {

				for (size_t x = 0; x < spinImageWidthPixels; x++) {
					for (size_t y = 0; y < spinImageWidthPixels; y++) {
						spinPixelType redPixelValue = redChannelDescriptors.content[imageIndex].contents[size_t(spinImageWidthPixels) * y + x];
                        spinPixelType greenPixelValue = greenChannelDescriptors.content[imageIndex].contents[size_t(spinImageWidthPixels) * y + x];
                        spinPixelType bluePixelValue = blueChannelDescriptors.content[imageIndex].contents[size_t(spinImageWidthPixels) * y + x];

                        unsigned char redPixelByte = computeImageByte(logarithmicImage, redMax, redPixelValue);
                        unsigned char greenPixelByte = computeImageByte(logarithmicImage, greenMax, greenPixelValue);
                        unsigned char bluePixelByte = computeImageByte(logarithmicImage, blueMax, bluePixelValue);

                        size_t pixelX = col * (spinImageWidthPixels + 1) + x;
						size_t pixelY = row * (spinImageWidthPixels + 1) +
										      (spinImageWidthPixels - 1 - y); // Flip image because image coordinates

						size_t pixelBaseIndex = 4 * (pixelX + width * pixelY);

						if(!std::isnan((float) redPixelByte) && !std::isnan((float) greenPixelByte) && !std::isnan((float) bluePixelByte)) {
							imageData[pixelBaseIndex + 0] = redPixelByte;
							imageData[pixelBaseIndex + 1] = greenPixelByte;
							imageData[pixelBaseIndex + 2] = bluePixelByte;
							imageData[pixelBaseIndex + 3] = 255;
						} else {
							imageData[pixelBaseIndex + 0] = 255;
							imageData[pixelBaseIndex + 1] = 0;
							imageData[pixelBaseIndex + 2] = 0;
							imageData[pixelBaseIndex + 3] = 255;
						}
					}
				}
			}

			imageIndex++;
		}
	}

	std::cout << "Writing image file.. " << imageDestinationFile << std::endl;

	unsigned error = lodepng::encode(imageDestinationFile.string(), imageData, width, height);

	if(error)
	{
		std::cout << "encoder error " << error << ": " << lodepng_error_text(error) << std::endl;
	}
}

void ShapeDescriptor::dump::descriptors(ShapeDescriptor::cpu::array<ShapeDescriptor::SpinImageDescriptor> hostDescriptors, std::experimental::filesystem::path imageDestinationFile, bool logarithmicImage, unsigned int imagesPerRow)
{
	performSpinDump<float, ShapeDescriptor::SpinImageDescriptor>(hostDescriptors, hostDescriptors, hostDescriptors, imageDestinationFile, logarithmicImage, imagesPerRow);
}

void ShapeDescriptor::dump::descriptors(ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor> hostDescriptors, std::experimental::filesystem::path imageDestinationFile, bool logarithmicImage, unsigned int imagesPerRow)
{
	performSpinDump<unsigned int, ShapeDescriptor::RICIDescriptor> (hostDescriptors, hostDescriptors, hostDescriptors, imageDestinationFile, logarithmicImage, imagesPerRow);
}

void ShapeDescriptor::dump::descriptors(ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> hostDescriptors, std::experimental::filesystem::path imageDestinationFile, unsigned int imagesPerRow) {
    descriptorComparisonImage(imageDestinationFile, hostDescriptors, hostDescriptors, hostDescriptors, imagesPerRow);
}

void ShapeDescriptor::dump::descriptorComparisonImage(std::experimental::filesystem::path imageDestinationFile,
                                                      ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> blueChannelDescriptors,
                                                      ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> greenChannelDescriptors,
                                                      ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> redChannelDescriptors,
                                                      unsigned int imagesPerRow) {

    unsigned int redCount = redChannelDescriptors.length;
    unsigned int greenCount = greenChannelDescriptors.length;
    unsigned int blueCount = blueChannelDescriptors.length;
    unsigned int totalDescriptorCount = std::max(redCount, std::max(greenCount, blueCount));
    bool redCountValid = redChannelDescriptors.content == nullptr || redCount == totalDescriptorCount;
    bool greenCountValid = greenChannelDescriptors.content == nullptr || greenCount == totalDescriptorCount;
    bool blueCountValid = blueChannelDescriptors.content == nullptr || blueCount == totalDescriptorCount;
    if(!redCountValid || !greenCountValid || !blueCountValid) {
        std::cerr << "For this image type you must provide arrays with equal numbers of images. Make sure the array lengths are the same!" << std::endl;
    }

    unsigned int descriptorCount = blueChannelDescriptors.length;

    // Compute the number of images that should be inserted to separate the two series
    // If the number of rows fits the images exactly, an extra one is inserted for better clarity.
    size_t rowRemainder = descriptorCount % imagesPerRow;
    size_t fillerImageCount = (rowRemainder == 0) ? 0 : (imagesPerRow - rowRemainder);

    size_t totalImageCount = descriptorCount + fillerImageCount;

    ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor> redDecompressedDescriptors(totalImageCount);
    ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor> blueDecompressedDescriptors(totalImageCount);
    ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor> greenDecompressedDescriptors(totalImageCount);


    for(unsigned int imageIndex = 0; imageIndex < descriptorCount; imageIndex++) {
        for(unsigned int chunkIndex = 0; chunkIndex < UINTS_PER_QUICCI; chunkIndex++) {
            unsigned int redChunk = redChannelDescriptors.content == nullptr ? 0 : redChannelDescriptors.content[imageIndex].contents[chunkIndex];
            unsigned int blueChunk = blueChannelDescriptors.content == nullptr ? 0 : blueChannelDescriptors.content[imageIndex].contents[chunkIndex];
            unsigned int greenChunk = greenChannelDescriptors.content == nullptr ? 0 : greenChannelDescriptors.content[imageIndex].contents[chunkIndex];
            std::bitset<32> redEntryBits(redChunk);
            std::bitset<32> greenEntryBits(greenChunk);
            std::bitset<32> blueEntryBits(blueChunk);
            for(char bit = 0; bit < 32; bit++) {
                redDecompressedDescriptors.content[imageIndex].contents[32 * chunkIndex + bit] = unsigned(int(redEntryBits[31 - bit]) * 255);
                greenDecompressedDescriptors.content[imageIndex].contents[32 * chunkIndex + bit] = unsigned(int(greenEntryBits[31 - bit]) * 255);
                blueDecompressedDescriptors.content[imageIndex].contents[32 * chunkIndex + bit] = unsigned(int(blueEntryBits[31 - bit]) * 255);
            }
        }
    }

    unsigned int pixelIndex = descriptorCount * UINTS_PER_QUICCI * 32;

    for(unsigned int emptyImageIndex = descriptorCount; emptyImageIndex < totalImageCount; emptyImageIndex++) {
        for(int i = 0; i < spinImageWidthPixels * spinImageWidthPixels; i++) {
            redDecompressedDescriptors.content[emptyImageIndex].contents[i] = UINT32_MAX;
            greenDecompressedDescriptors.content[emptyImageIndex].contents[i] = UINT32_MAX;
            blueDecompressedDescriptors.content[emptyImageIndex].contents[i] = UINT32_MAX;
            pixelIndex++;
        }
    }

    performSpinDump<unsigned int, ShapeDescriptor::RICIDescriptor>(redDecompressedDescriptors,
                                                                   greenDecompressedDescriptors,
                                                                   blueDecompressedDescriptors,
                                                                   imageDestinationFile,
                                                                   false,
                                                                   imagesPerRow);

    ShapeDescriptor::free::array(redDecompressedDescriptors);
    ShapeDescriptor::free::array(blueDecompressedDescriptors);
    ShapeDescriptor::free::array(greenDecompressedDescriptors);
}
