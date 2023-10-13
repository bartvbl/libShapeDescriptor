#include <glm/vec3.hpp>
#include <glm/geometric.hpp>
#include <shapeDescriptor/shapeDescriptor.h>

#pragma GCC optimize ("0")

ShapeDescriptor::cpu::float3
ShapeDescriptor::computeTriangleNormal(
        ShapeDescriptor::cpu::float3 &triangleVertex0,
        ShapeDescriptor::cpu::float3 &triangleVertex1,
        ShapeDescriptor::cpu::float3 &triangleVertex2) {
    ShapeDescriptor::cpu::float3 side0 = triangleVertex1 - triangleVertex0;
    ShapeDescriptor::cpu::float3 side1 = triangleVertex2 - triangleVertex0;


    side0 = side0 / length(side0);
    side1 = side1 / length(side1);

    glm::vec3 glmSide0NonNormalised = glm::vec3(side0.x, side0.y, side0.z);
    glm::vec3 glmSide1NonNormalised = glm::vec3(side1.x, side1.y, side1.z);

    glm::vec3 glmSide0 = glmSide0NonNormalised;
    glm::vec3 glmSide1 = glmSide1NonNormalised;

    glm::vec3 glmNormal = glm::cross(glmSide0, glmSide1);

    float length = glm::length(glmNormal);

    if(length != 0) {
        glmNormal.x /= length;
        glmNormal.y /= length;
        glmNormal.z /= length;
    } else {
        // Some objects may have zero-area triangles. In that case, we use an arbitrarily chosen fallback normal
        glmNormal = {0, 0, 1};
    }



    // GIVES INCORRECT RESULTS (0, -0.76, 0) -> (-1, 0, 0) for SOME reason
//glmNormal = glm::normalize(glmNormal);

    ShapeDescriptor::cpu::float3 normal = make_float3_cpu(glmNormal.x, glmNormal.y, glmNormal.z);


    return normal;
}

ShapeDescriptor::cpu::float3 hostComputeTriangleNormal(std::vector<ShapeDescriptor::cpu::float3> &vertices, unsigned int baseIndex) {
    ShapeDescriptor::cpu::float3 triangleVertex0 = vertices.at(baseIndex + 0);
    ShapeDescriptor::cpu::float3 triangleVertex1 = vertices.at(baseIndex + 1);
    ShapeDescriptor::cpu::float3 triangleVertex2 = vertices.at(baseIndex + 2);

    return ShapeDescriptor::computeTriangleNormal(triangleVertex0, triangleVertex1, triangleVertex2);
}
#pragma GCC reset_options