#include <spinImage/libraryBuildSettings.h>
#include <spinImage/cpu/types/CPURasterisationSettings.h>
#include "MSIGenerator.h"
#include "QSIGenerator.h"

void SpinImage::cpu::computeMSIFallingHorizontal(array<unsigned int> MSIDescriptor, array<unsigned int> QSIDescriptor) {
    for (int y = 0; y < spinImageWidthPixels; y++)
    {
        for (int x = 0; x < spinImageWidthPixels - 1; x++)
        {
            MSIDescriptor.content[y * spinImageWidthPixels + x] = QSIDescriptor.content[y * spinImageWidthPixels + x] > QSIDescriptor.content[y * spinImageWidthPixels + x + 1] ? 1 : 0;
        }
    }
}

void SpinImage::cpu::computeMSIRisingHorizontal(array<unsigned int> MSIDescriptor, array<unsigned int> QSIDescriptor) {
    for (int y = 0; y < spinImageWidthPixels; y++)
    {
        for (int x = 0; x < spinImageWidthPixels - 1; x++)
        {
            MSIDescriptor.content[y * spinImageWidthPixels + x] = QSIDescriptor.content[y * spinImageWidthPixels + x] < QSIDescriptor.content[y * spinImageWidthPixels + x + 1] ? 1 : 0;
        }
    }
}
