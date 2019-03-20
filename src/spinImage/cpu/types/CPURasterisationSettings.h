#pragma once

#include "HostMesh.h"
#include "float3_cpu.h"

struct CPURasterisationSettings {
    float3_cpu spinImageVertex;
    float3_cpu spinImageNormal;
    int vertexIndexIndex;

    HostMesh mesh;
};