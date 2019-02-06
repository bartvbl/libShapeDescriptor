#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <limits>

#include "shapeSearch/common/geom.hpp"
#include "shapeSearch/cpu/hostMesh.h"

HostMesh hostLoadOBJ(std::string src, MeshFormat expectedOutputFormat, bool recomputeNormals = false);