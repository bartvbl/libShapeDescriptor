#pragma once

#include <algorithm>
#include <shapeSearch/cpu/types/CPURasterisationSettings.h>
#include <shapeSearch/cpu/types/float3_cpu.h>

#include "shapeSearch/utilities/OBJLoader.h"
#include "shapeSearch/common/types/array.h"

void hostGenerateQSI(array<unsigned int> descriptor, CPURasterisationSettings settings);
array<unsigned int> hostGenerateQSIAllVertices(CPURasterisationSettings settings);

// utility functions
float3_cpu hostTransformCoordinate(float3_cpu vertex, float3_cpu spinImageVertex, float3_cpu spinImageNormal);



