#pragma once

#include <algorithm>
#include <shapeSearch/cpu/types/CPURasterisationSettings.h>
#include <shapeSearch/cpu/types/float3_cpu.h>

#include "shapeSearch/utilities/OBJLoader.h"
#include "shapeSearch/common/types/array.h"

void hostGenerateQSI(array<quasiSpinImagePixelType> descriptor, CPURasterisationSettings settings);
array<quasiSpinImagePixelType> hostGenerateQSIAllVertices(CPURasterisationSettings settings);