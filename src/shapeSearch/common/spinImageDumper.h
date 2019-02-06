#pragma once
#include "cudaCommon.h"
#include "lodepng.h"

void dumpImages(VertexDescriptors descriptors, OutputImageSettings imageSettings);
void dumpCompressedImages(array<unsigned int> compressedDescriptors, OutputImageSettings imageSettings);
void dumpRawCompressedImages(array<unsigned int> compressedDescriptors, std::string destination);