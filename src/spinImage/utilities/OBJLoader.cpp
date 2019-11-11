#include "OBJLoader.h"
#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>

#include <glm/vec3.hpp>
#include <glm/geometric.hpp>

SpinImage::cpu::float3 hostComputeTriangleNormal(std::vector<SpinImage::cpu::float3> &vertices, unsigned int baseIndex);

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

inline SpinImage::cpu::float3 elementWiseMin(SpinImage::cpu::float3 v1, SpinImage::cpu::float3 v2)
{
	SpinImage::cpu::float3 output;
	output.x = std::min(v1.x, v2.x);
	output.y = std::min(v1.y, v2.y);
	output.z = std::min(v1.z, v2.z);
	return output;
}

inline SpinImage::cpu::float3 elementWiseMax(SpinImage::cpu::float3 v1, SpinImage::cpu::float3 v2)
{
	SpinImage::cpu::float3 output;
	output.x = std::max(v1.x, v2.x);
	output.y = std::max(v1.y, v2.y);
	output.z = std::max(v1.z, v2.z);
	return output;
}

SpinImage::cpu::Mesh SpinImage::utilities::loadOBJ(std::string src, bool recomputeNormals)
{
	std::vector<std::string> lineParts;
	lineParts.reserve(32);
	std::vector<std::string> faceParts;
	faceParts.reserve(32);
	
	std::vector<cpu::float3> vertices;
	std::vector<cpu::float3> normals;
	std::vector<cpu::float2> textureCoordinates;

	std::vector<cpu::float3> vertexBuffer;
	std::vector<cpu::float2> textureBuffer;
	std::vector<cpu::float3> normalBuffer;

	cpu::float3 boundingBoxMin;
	cpu::float3 boundingBoxMax;

	std::vector<unsigned int> indices;

	std::ifstream objFile(src);
	std::string line;

	unsigned int currentIndex = 0;

	if (objFile.is_open()) {
		while (std::getline(objFile, line)) {
			lineParts.clear();
			split(&lineParts, line, ' ');
			deleteEmptyStrings(lineParts);

			if (lineParts.size() == 0) {
				continue;
			}

			if (lineParts.at(0) == "v") {
				cpu::float3 vertex;
				vertex.x = std::stof(lineParts.at(1));
				vertex.y = std::stof(lineParts.at(2));
				vertex.z = std::stof(lineParts.at(3));
				vertexBuffer.push_back(vertex);
			}

			if (lineParts.at(0) == "vn") {
				cpu::float3 normal;
				normal.x = std::stof(lineParts.at(1));
				normal.y = std::stof(lineParts.at(2));
				normal.z = std::stof(lineParts.at(3));
				normalBuffer.push_back(normal);
			}

			if (lineParts.at(0) == "vt") {
				cpu::float2 textureCoordinate;
				textureCoordinate.x = std::stof(lineParts.at(1));
				textureCoordinate.y = std::stof(lineParts.at(2));
				textureBuffer.push_back(textureCoordinate);
			}

			if (lineParts.at(0) == "f") {
				bool normalsFound = false;
				for (int i = 1; i <= 3; i++) {
					faceParts.clear();
					std::string linePart = lineParts.at(i);
					split(&faceParts, linePart, '/');

					int vertexIndex = std::stoi(faceParts.at(0)) - 1;
					cpu::float3 vertex = vertexBuffer.at(unsigned(vertexIndex));
					vertices.push_back(vertex);

					if(currentIndex == 0) {
						boundingBoxMin = vertex;
						boundingBoxMax = vertex;
					} else {
						boundingBoxMin = elementWiseMin(boundingBoxMin, vertex);
						boundingBoxMax = elementWiseMax(boundingBoxMax, vertex);
					}

					int normalIndex = -1;

					if(faceParts.size() == 2)
					{
						normalIndex = std::stoi(faceParts.at(1)) - 1;
					}
					else if (faceParts.size() == 3 && faceParts.at(1) == "") {
						normalIndex = std::stoi(faceParts.at(2)) - 1;
					}
					else if (faceParts.size() == 3)
					{
						int textureCoordIndex = std::stoi(faceParts.at(1)) - 1;
					    if(textureBuffer.size() > textureCoordIndex) {
                            textureCoordinates.push_back(textureBuffer.at(textureCoordIndex));
                        // This situation can occur when the file format is either invalid
                        // (does not include texture coordinates at all)
                        // or has not specified the texture coordinates yet (something this loader does not deal with)
					    } else {
					        textureCoordinates.push_back(make_float2_cpu(0, 0));
					    }
						normalIndex = std::stoi(faceParts.at(2)) - 1;
					}

					// (partially) invalid files may contain normals later, or in some cases not at all.
					// This check accounts for that. If it fails, normals are computed after processing
					// the contents of the current line in the file
					if(normalIndex != -1 && normalBuffer.size() > normalIndex) {
						normals.push_back(normalBuffer.at(normalIndex));
						normalsFound = true;
					}

					indices.push_back(currentIndex);
					currentIndex++;
				}

				// If the file incorrectly or was missing normals, we compute them here.
				// Alternatively, we override those present in the file if this was mandated by the use.
				if(!normalsFound) {
                    cpu::float3 normal = hostComputeTriangleNormal(vertices, vertices.size() - 3);

                    normals.push_back(normal);
                    normals.push_back(normal);
                    normals.push_back(normal);
				}
			}
		}

        if(recomputeNormals) {
            for(int index = 0; index < indices.size(); index+=3) {
                cpu::float3 recomputedNormal = hostComputeTriangleNormal(vertices, index);
                normals.at(index + 0) = recomputedNormal;
                normals.at(index + 1) = recomputedNormal;
                normals.at(index + 2) = recomputedNormal;
            }
        }

        unsigned int faceCount = unsigned(indices.size()) / 3;

		cpu::float3* meshVertexBuffer = new cpu::float3[vertices.size()];
		std::copy(vertices.begin(), vertices.end(), meshVertexBuffer);

		cpu::float3* meshNormalBuffer = new cpu::float3[normals.size()];
		std::copy(normals.begin(), normals.end(), meshNormalBuffer);

		cpu::float2* meshTextureCoordBuffer = new cpu::float2[textureCoordinates.size()];
		std::copy(textureCoordinates.begin(), textureCoordinates.end(), meshTextureCoordBuffer);

		unsigned int* meshIndexBuffer = new unsigned int[3 * faceCount];
		std::copy(indices.begin(), indices.end(), meshIndexBuffer);

		objFile.close();

		cpu::Mesh mesh;

		mesh.vertices = meshVertexBuffer;
		mesh.normals = meshNormalBuffer;

		mesh.indices = meshIndexBuffer;

		mesh.vertexCount = 3 * faceCount;
		mesh.indexCount = 3 * faceCount;

		//mesh.boundingBoxMin = boundingBoxMin;
		//mesh.boundingBoxMax = boundingBoxMax;

		return mesh;
	}
	else {
		std::cout << "OBJ file at " << src << " failed to load!" << std::endl;
	}

	return cpu::Mesh();
}

SpinImage::cpu::float3 hostComputeTriangleNormal(std::vector<SpinImage::cpu::float3> &vertices, unsigned int baseIndex) {
    SpinImage::cpu::float3 triangleVertex0 = vertices.at(baseIndex + 0);
    SpinImage::cpu::float3 triangleVertex1 = vertices.at(baseIndex + 1);
    SpinImage::cpu::float3 triangleVertex2 = vertices.at(baseIndex + 2);

    SpinImage::cpu::float3 side0 = triangleVertex1 - triangleVertex0;
    SpinImage::cpu::float3 side1 = triangleVertex2 - triangleVertex0;


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

    SpinImage::cpu::float3 normal = make_float3_cpu(glmNormal.x, glmNormal.y, glmNormal.z);

    return normal;
}
