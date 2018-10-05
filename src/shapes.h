#pragma once

#include "Mesh.h"

Mesh generateSphere();
Mesh generateCylinder(float3 orgin, float3 direction, float radius, float height, unsigned int numSlices);