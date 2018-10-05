#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <limits>

#include "shapeSearch/geom.hpp"
#include "shapeSearch/Mesh.h"

Mesh loadOBJ(std::string src, MeshFormat expectedOutputFormat, bool recomputeNormals = false);