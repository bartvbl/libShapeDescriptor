#include <shapeDescriptor/shapeDescriptor.h>
#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>

#include <glm/vec3.hpp>
#include <glm/geometric.hpp>
#include <fast-obj/fast_obj.h>
#include <algorithm>
#include <filesystem>

void split(std::vector<std::string>* parts, const std::string &s, char delim) {
	
	std::stringstream ss;
	ss.str(s);
	std::string item;
	while (std::getline(ss, item, delim)) {
		parts->push_back(item);
	}
}

void deleteEmptyStrings(std::vector<std::string> &list) {
	for (int i = 0; i < list.size(); i++) {
		std::string item = list.at(i);
		if (item == "") {
			list.erase(list.begin() + i);
			i--;
		}
	}
}

inline ShapeDescriptor::cpu::float3 elementWiseMin(ShapeDescriptor::cpu::float3 v1, ShapeDescriptor::cpu::float3 v2)
{
	ShapeDescriptor::cpu::float3 output;
	output.x = std::min(v1.x, v2.x);
	output.y = std::min(v1.y, v2.y);
	output.z = std::min(v1.z, v2.z);
	return output;
}

inline ShapeDescriptor::cpu::float3 elementWiseMax(ShapeDescriptor::cpu::float3 v1, ShapeDescriptor::cpu::float3 v2)
{
	ShapeDescriptor::cpu::float3 output;
	output.x = std::max(v1.x, v2.x);
	output.y = std::max(v1.y, v2.y);
	output.z = std::max(v1.z, v2.z);
	return output;
}

ShapeDescriptor::cpu::Mesh ShapeDescriptor::loadOBJ(std::filesystem::path src, RecomputeNormals recomputeNormals)
{
    fastObjMesh* temporaryMesh = fast_obj_read(src.c_str());

    unsigned int faceCount = 0;

    for(unsigned int groupIndex = 0; groupIndex < temporaryMesh->group_count; groupIndex++) {
        faceCount += temporaryMesh->groups[groupIndex].face_count;
    }

    bool hasNormals = temporaryMesh->normal_count != 1;
    bool shouldRecomputeNormals = (!hasNormals && recomputeNormals == RecomputeNormals::RECOMPUTE_IF_MISSING) || recomputeNormals == RecomputeNormals::ALWAYS_RECOMPUTE;



    ShapeDescriptor::cpu::float3* meshVertexBuffer = new ShapeDescriptor::cpu::float3[3 * faceCount];
    ShapeDescriptor::cpu::float3* meshNormalBuffer = (hasNormals || shouldRecomputeNormals) ? new ShapeDescriptor::cpu::float3[3 * faceCount] : nullptr;

    unsigned int nextVertexIndex = 0;

    for(unsigned int groupIndex = 0; groupIndex < temporaryMesh->group_count; groupIndex++) {
        fastObjGroup group = temporaryMesh->groups[groupIndex];

        for (unsigned int faceIndex = 0; faceIndex < group.face_count; faceIndex++) {
            // Ensure faces are triangles. Can probably fix this at a later date, but for now it gives an error.
            unsigned int verticesPerFace = temporaryMesh->face_vertices[faceIndex + group.face_offset];
            if (verticesPerFace != 3) {
                throw std::runtime_error(
                        "This OBJ loader only supports 3 vertices per face. The model file " + src.string() +
                        " contains a face with " + std::to_string(verticesPerFace) + " vertices.\n"
                        "You can usually solve this problem by re-exporting the object from a 3D model editor, "
                        "and selecting the OBJ export option that forces triangle faces.");
            }

            for (unsigned int i = 0; i < 3; i++) {
                fastObjIndex index = temporaryMesh->indices[3 * faceIndex + i + group.index_offset];

                meshVertexBuffer[nextVertexIndex] = {
                        temporaryMesh->positions[3 * index.p + 0],
                        temporaryMesh->positions[3 * index.p + 1],
                        temporaryMesh->positions[3 * index.p + 2]};

                if (hasNormals && !shouldRecomputeNormals) {
                    meshNormalBuffer[nextVertexIndex] = {
                            temporaryMesh->normals[3 * index.n + 0],
                            temporaryMesh->normals[3 * index.n + 1],
                            temporaryMesh->normals[3 * index.n + 2]};
                }

                nextVertexIndex++;
            }

            if(shouldRecomputeNormals) {
                ShapeDescriptor::cpu::float3 normal = computeTriangleNormal(
                        meshVertexBuffer[nextVertexIndex - 3],
                        meshVertexBuffer[nextVertexIndex - 2],
                        meshVertexBuffer[nextVertexIndex - 1]);

                meshNormalBuffer[nextVertexIndex - 3] = normal;
                meshNormalBuffer[nextVertexIndex - 2] = normal;
                meshNormalBuffer[nextVertexIndex - 1] = normal;
            }
        }
    }

    ShapeDescriptor::cpu::Mesh mesh;

    mesh.vertices = meshVertexBuffer;
    mesh.normals = meshNormalBuffer;

    mesh.vertexCount = 3 * faceCount;

    fast_obj_destroy(temporaryMesh);

    return mesh;
}

