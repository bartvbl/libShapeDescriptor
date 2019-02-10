#pragma once

#include <shapeSearch/common/types/array.h>

void hostComputeMSI_risingHorizontal(array<unsigned int> MSIDescriptor, array<unsigned int> QSIDescriptor);
void hostComputeMSI_fallingHorizontal(array<unsigned int> MSIDescriptor, array<unsigned int> QSIDescriptor);