#include <cassert>
#include <cmath>
#include <limits>
#include "MeshScaler.h"

SpinImage::cpu::Mesh SpinImage::utilities::scaleMesh(cpu::Mesh &model, cpu::Mesh &scaledModel, float spinImagePixelSize)
{
    assert(model.vertexCount == scaledModel.vertexCount);

    for (size_t i = 0; i < model.vertexCount; i++) {
        scaledModel.vertices[i].x = model.vertices[i].x / spinImagePixelSize;
        scaledModel.vertices[i].y = model.vertices[i].y / spinImagePixelSize;
        scaledModel.vertices[i].z = model.vertices[i].z / spinImagePixelSize;
    }
}


SpinImage::cpu::Mesh SpinImage::utilities::fitMeshInsideSphereOfRadius(SpinImage::cpu::Mesh &input, float radius) {
    double averageX = 0;
    double averageY = 0;
    double averageZ = 0;

    // I use a running average mean computing method here for better accuracy with large models
    for(unsigned int i = 0; i < input.vertexCount; i++) {
        SpinImage::cpu::float3 vertex = input.vertices[i];

        averageX += (vertex.x - averageX) / float(i + 1);
        averageY += (vertex.y - averageY) / float(i + 1);
        averageZ += (vertex.z - averageZ) / float(i + 1);
    }

    double maxDistance = -std::numeric_limits<double>::max();

    for(unsigned int i = 0; i < input.vertexCount; i++) {
        SpinImage::cpu::float3 vertex = input.vertices[i];

        double deltaX = vertex.x - averageX;
        double deltaY = vertex.y - averageY;
        double deltaZ = vertex.z - averageZ;

        double length = std::sqrt(deltaX * deltaX + deltaY * deltaY + deltaZ * deltaZ);
        maxDistance = std::max(maxDistance, length);
    }


    SpinImage::cpu::Mesh scaledMesh(input.vertexCount, input.indexCount);

    std::copy(input.normals, input.normals + scaledMesh.vertexCount, scaledMesh.normals);
    std::copy(input.indices, input.indices + scaledMesh.indexCount, scaledMesh.indices);

    double scaleFactor = (radius / maxDistance);

    for(unsigned int i = 0; i < input.vertexCount; i++) {
        scaledMesh.vertices[i].x = float((double(input.vertices[i].x) - averageX) * scaleFactor);
        scaledMesh.vertices[i].y = float((double(input.vertices[i].y) - averageY) * scaleFactor);
        scaledMesh.vertices[i].z = float((double(input.vertices[i].z) - averageZ) * scaleFactor);
    }

    return scaledMesh;
}