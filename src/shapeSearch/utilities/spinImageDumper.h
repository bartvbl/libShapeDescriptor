#pragma once

#include <shapeSearch/common/types/OutputImageSettings.h>
#include <shapeSearch/common/types/array.h>
#include <shapeSearch/common/types/VertexDescriptors.h>
#include <shapeSearch/libraryBuildSettings.h>

#include <string>

void dumpImages(VertexDescriptors descriptors, OutputImageSettings imageSettings, unsigned int imagesPerRow);
void dumpCompressedImages(array<unsigned int> compressedDescriptors, OutputImageSettings imageSettings, unsigned int imagesPerRow);
void dumpRawCompressedImages(array<unsigned int> compressedDescriptors, std::string destination, unsigned int imagesPerRow);