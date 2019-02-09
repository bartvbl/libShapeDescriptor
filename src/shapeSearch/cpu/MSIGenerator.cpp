#include <shapeSearch/libraryBuildSettings.h>
#include <shapeSearch/cpu/types/CPURasterisationSettings.h>
#include "MSIGenerator.h"
#include "QSIGenerator.hpp"

void hostGenerateMSI(array<unsigned int> MSIDescriptor, array<unsigned int> QSIDescriptor, CPURasterisationSettings settings) {
    hostGenerateQSI(QSIDescriptor, settings);
}

void hostComputeMSI_fallingHorizontal(array<unsigned int> MSIDescriptor, array<unsigned int> QSIDescriptor) {
    for (int y = 0; y < spinImageWidthPixels; y++)
    {
        for (int x = 0; x < spinImageWidthPixels - 1; x++)
        {
            MSIDescriptor.content[y * spinImageWidthPixels + x] = QSIDescriptor.content[y * spinImageWidthPixels + x] > QSIDescriptor.content[y * spinImageWidthPixels + x + 1] ? 1 : 0;
        }
    }
}

void hostComputeMSI_risingHorizontal(array<unsigned int> MSIDescriptor, array<unsigned int> QSIDescriptor) {
    for (int y = 0; y < spinImageWidthPixels; y++)
    {
        for (int x = 0; x < spinImageWidthPixels - 1; x++)
        {
            MSIDescriptor.content[y * spinImageWidthPixels + x] = QSIDescriptor.content[y * spinImageWidthPixels + x] < QSIDescriptor.content[y * spinImageWidthPixels + x + 1] ? 1 : 0;
        }
    }
}
