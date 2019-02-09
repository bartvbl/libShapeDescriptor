#pragma once

#include <shapeSearch/common/types/array.h>
#include <shapeSearch/common/types/VertexDescriptors.h>
#include <shapeSearch/libraryBuildSettings.h>

#include <string>

void dumpImages(VertexDescriptors descriptors, std::string imageDestinationFile, bool logarithmicImage, unsigned int imagesPerRow);
void dumpCompressedImages(array<unsigned int> compressedDescriptors, std::string imageDestinationFile, bool logarithmicImage, unsigned int imagesPerRow);
void dumpRawCompressedImages(array<unsigned int> compressedDescriptors, std::string destination, unsigned int imagesPerRow);