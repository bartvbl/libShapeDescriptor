#include <shapeDescriptor/shapeDescriptor.h>
#include <unordered_set>
#include <vector>
#include <unordered_map>

ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint>
ShapeDescriptor::generateSpinOriginBuffer(const ShapeDescriptor::cpu::Mesh &mesh) {
    ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> originBuffer(mesh.vertexCount);
    for(size_t i = 0; i < mesh.vertexCount; i++) {
        originBuffer.content[i] = ShapeDescriptor::OrientedPoint{mesh.vertices[i], mesh.normals[i]};
    }
    return originBuffer;
}

ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint>
ShapeDescriptor::generateUniqueSpinOriginBuffer(const ShapeDescriptor::cpu::Mesh &mesh, std::vector<size_t>* mapping) {
    std::unordered_map<ShapeDescriptor::OrientedPoint, size_t> indexMapping;
    if(mapping != nullptr) {
        mapping->resize(mesh.vertexCount);
    }
    std::vector<ShapeDescriptor::OrientedPoint> originBuffer;
    originBuffer.reserve(mesh.vertexCount);
    std::unordered_set<ShapeDescriptor::OrientedPoint> seenSet;
    for(size_t i = 0; i < mesh.vertexCount; i++) {
        ShapeDescriptor::OrientedPoint currentPoint = ShapeDescriptor::OrientedPoint{mesh.vertices[i], mesh.normals[i]};
        if(seenSet.count(currentPoint) == 0) {
            seenSet.insert(currentPoint);
            originBuffer.emplace_back(currentPoint);
            if(mapping != nullptr) {
                mapping->at(i) = originBuffer.size() - 1;
                indexMapping.insert({currentPoint, originBuffer.size()});
            }
        } else {
            if(mapping != nullptr) {
                mapping->at(i) = indexMapping.at(currentPoint);
            }
        }
    }

    ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> outputBuffer(originBuffer.size());
    std::copy(originBuffer.begin(), originBuffer.end(), outputBuffer.content);
    return outputBuffer;
}
