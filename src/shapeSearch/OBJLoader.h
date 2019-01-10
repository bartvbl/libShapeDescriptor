#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <limits>

#include "shapeSearch/geom.hpp"
#include "shapeSearch/Mesh.h"

HostMesh loadOBJ(std::string src, MeshFormat expectedOutputFormat, bool recomputeNormals = false);