#pragma once

#include "shapeSearch/cpu/types/HostMesh.h"

HostMesh hostLoadOBJ(std::string src, MeshFormat expectedOutputFormat, bool recomputeNormals = false);