#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <limits>

#include "shapeSearch/cpu/geom.hpp"
#include "shapeSearch/cpu/Mesh.h"

HostMesh hostLoadOBJ(std::string src, MeshFormat expectedOutputFormat, bool recomputeNormals = false);