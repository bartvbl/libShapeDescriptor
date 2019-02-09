#include "MeshScaler.h"

HostMesh hostScaleMesh(HostMesh &model, HostMesh &scaledModel, float spinImagePixelSize)
{
    scaledModel.indexCount = model.indexCount;
    scaledModel.indices = model.indices;
    scaledModel.normals = model.normals;
    scaledModel.vertexCount = model.vertexCount;

    scaledModel.vertices = new float3_cpu[model.vertexCount];

    for (int i = 0; i < model.vertexCount; i++) {
        scaledModel.vertices[i].x = model.vertices[i].x / spinImagePixelSize;
        scaledModel.vertices[i].y = model.vertices[i].y / spinImagePixelSize;
        scaledModel.vertices[i].z = model.vertices[i].z / spinImagePixelSize;
    }

    return model;
}