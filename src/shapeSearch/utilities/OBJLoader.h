#pragma once

#include "shapeSearch/cpu/types/HostMesh.h"

HostMesh hostLoadOBJ(std::string src, bool recomputeNormals = false);