#include <assert.h>
#include "MeshScaler.h"

HostMesh hostScaleMesh(HostMesh &model, HostMesh &scaledModel, float spinImagePixelSize)
{
    assert(model.vertexCount == scaledModel.vertexCount);

    for (int i = 0; i < model.vertexCount; i++) {
        scaledModel.vertices[i].x = model.vertices[i].x / spinImagePixelSize;
        scaledModel.vertices[i].y = model.vertices[i].y / spinImagePixelSize;
        scaledModel.vertices[i].z = model.vertices[i].z / spinImagePixelSize;
    }
}