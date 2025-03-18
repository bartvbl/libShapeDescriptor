#include <iostream>
#include <shapeDescriptor/shapeDescriptor.h>

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
// #define TINYGLTF_NOEXCEPTION // optional. disable exception handling.
#include "tiny_gltf.h"
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_inverse.hpp>

const std::array<std::string, 7> gltfDrawModes = {"POINTS", "LINES", "LINE_LOOP", "LINE_STRIP", "TRIANGLES", "TRIANGLE_STRIP", "TRIANGLE_FAN"};
enum class GLTFDrawMode {
    POINTS = 0,
    LINES = 1,
    LINE_LOOP = 2,
    LINE_STRIP = 3,
    TRIANGLES = 4,
    TRIANGLE_STRIP = 5,
    TRIANGLE_FAN = 6,
    END = 7
};

enum class UnsupportedDrawModeBehaviour {
    IGNORE_AND_EXCLUDE, THROW_ERROR
};

const UnsupportedDrawModeBehaviour meshIncludesLinesBehaviour = UnsupportedDrawModeBehaviour::IGNORE_AND_EXCLUDE;
const UnsupportedDrawModeBehaviour meshIncludesPointsBehaviour = UnsupportedDrawModeBehaviour::THROW_ERROR;
const UnsupportedDrawModeBehaviour meshIncludesTriangleStripOrFanBehaviour = UnsupportedDrawModeBehaviour::THROW_ERROR;

const UnsupportedDrawModeBehaviour pointCloudIncludesTrianglesBehaviour = UnsupportedDrawModeBehaviour::IGNORE_AND_EXCLUDE;
const UnsupportedDrawModeBehaviour pointCloudIncludesLinesBehaviour = UnsupportedDrawModeBehaviour::IGNORE_AND_EXCLUDE;
const UnsupportedDrawModeBehaviour pointCloudIncludesTrianglesStripOrFanBehaviour = UnsupportedDrawModeBehaviour::IGNORE_AND_EXCLUDE;

void reportDrawModeError(const std::filesystem::path &filePath, int drawMode) {
    throw std::runtime_error("The file loaded from " + filePath.string() + " contains geometry with an unsupported drawing mode (" + gltfDrawModes.at(drawMode) + "). Please re-export the object to use triangles exclusively, or use an alternate format.");
}

bool ShapeDescriptor::gltfContainsPointCloud(const std::filesystem::path& file) {
    std::ifstream inputStream{file};

    std::array<unsigned int, 3> fileHeader {0, 0, 0};
    inputStream.read((char*)fileHeader.data(), sizeof(fileHeader));


    nlohmann::json jsonHeader;

    if(fileHeader.at(0) == 0x46546C67) {
        assert(fileHeader.at(1) == 2); // GLTF revision should be 2
        //unsigned int totalSize = fileHeader.at(2);

        unsigned int headerChunkLength;
        unsigned int ignored_headerChunkType;
        inputStream.read((char*) &headerChunkLength, sizeof(unsigned int));
        inputStream.read((char*) &ignored_headerChunkType, sizeof(unsigned int));

        std::string jsonChunkContents;
        jsonChunkContents.resize(headerChunkLength);
        inputStream.read(jsonChunkContents.data(), headerChunkLength);
        jsonHeader = nlohmann::json::parse(jsonChunkContents);
    } else {
        // Reset stream
        inputStream.seekg(0);

        // The whole file is JSON, so we just need to read all of it
        jsonHeader = nlohmann::json::parse(inputStream);
    }

    for(const nlohmann::json& meshElement : jsonHeader.at("meshes")) {
        for(const nlohmann::json& primitive : meshElement.at("primitives")) {
            if(primitive.at("mode") == TINYGLTF_MODE_POINTS) {
                return true;
            }
        }
    }

    return false;
}

uint8_t scale16BitColourTo8Bit(uint16_t colour) {
    double floatColour = double(colour) / 65535.0;
    return uint8_t(floatColour * 255.0);
}

tinygltf::Model readTinyGLTFFile(const std::filesystem::path& path) {
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string binaryParseErrorMessage;
    std::string binaryParseWarningMessage;
    std::string asciiParseErrorMessage;
    std::string asciiParseWarningMessage;
    bool binarySuccess = true;
    bool asciiSuccess = true;

    // First try to read the file as a binary file. If that fails, try
    binarySuccess = loader.LoadBinaryFromFile(&model, &binaryParseErrorMessage, &binaryParseWarningMessage, path);
    if(!binarySuccess) {
        asciiSuccess = loader.LoadASCIIFromFile(&model, &asciiParseErrorMessage, &asciiParseWarningMessage, path);
    }

    // Binary load failed
    if(!binarySuccess && !asciiSuccess) {
        std::string errorMessage = "Failed to load GLTF file.";
        if(!binaryParseErrorMessage.empty()) {
            errorMessage += "\nErrors (as binary file): " + binaryParseErrorMessage;
        }
        if(!asciiParseErrorMessage.empty()) {
            errorMessage += "\nErrors (as ascii file): " + asciiParseErrorMessage;
        }
        if(!binaryParseWarningMessage.empty()) {
            errorMessage += "\nWarnings (as binary file): " + binaryParseWarningMessage;
        }
        if(!asciiParseWarningMessage.empty()) {
            errorMessage += "\nWarnings (as ascii file): " + asciiParseWarningMessage;
        }
        throw std::runtime_error(errorMessage);
    } else if(!binarySuccess && !asciiParseWarningMessage.empty()) {
        std::cout << "GLTF load warning: " + asciiParseWarningMessage << std::endl;
    } else if(binarySuccess && !binaryParseWarningMessage.empty()) {
        std::cout << "GLTF load warning: " + binaryParseWarningMessage << std::endl;
    }
    return model;
}

void computeMeshTransformation(const std::vector<tinygltf::Node> &nodes, int nodeIndex, glm::mat4 partialTransform, glm::mat3 partialNormalMatrix, std::vector<glm::mat4> &meshTransformationMatrices, std::vector<glm::mat3>& normalMatrices) {
    const tinygltf::Node& node = nodes.at(nodeIndex);

    glm::qua rotation = {1.0, 0.0, 0.0, 0.0};
    if(node.rotation.size() == 4) {
        // GLM uses a different format than GLTF
        rotation = {node.rotation.at(3), node.rotation.at(0), node.rotation.at(1), node.rotation.at(2)};
    }
    glm::mat4 rotationMatrix = glm::mat4_cast(rotation);

    glm::mat3 directionMatrix(1.0);
    directionMatrix[0] = {rotationMatrix[0].x, rotationMatrix[0].y, rotationMatrix[0].z};
    directionMatrix[1] = {rotationMatrix[1].x, rotationMatrix[1].y, rotationMatrix[1].z};
    directionMatrix[2] = {rotationMatrix[2].x, rotationMatrix[2].y, rotationMatrix[2].z};

    glm::vec3 translation = {0.0, 0.0, 0.0};
    if(node.translation.size() == 3) {
        translation = {node.translation.at(0), node.translation.at(1), node.translation.at(2)};
    }
    glm::mat4 translationMatrix = glm::translate(glm::mat4(1.0), translation);

    glm::vec3 scale = {1.0, 1.0, 1.0};
    if(node.scale.size() == 3) {
        scale = glm::vec3(node.scale.at(0), node.scale.at(1), node.scale.at(2));
    }
    glm::mat4 scaleMatrix = glm::scale(glm::mat4(1.0), scale);

    glm::mat4 transformationDefinedByNode(1.0);
    glm::mat3 normalMatrixDefinedByNode(1.0);
    if(node.matrix.size() == 16) {
        bool allZero = true;

        // Guaranteed to be in column major order
        for(uint32_t i = 0; i < node.matrix.size(); i++) {
            if(node.matrix.at(i) != 0) {
                allZero = false;
                break;
            }
        }
        if(allZero) {
            transformationDefinedByNode = glm::mat4(1.0);
        } else {
            transformationDefinedByNode[0] = {node.matrix.at(0), node.matrix.at(1), node.matrix.at(2), node.matrix.at(3)};
            transformationDefinedByNode[1] = {node.matrix.at(4), node.matrix.at(5), node.matrix.at(6), node.matrix.at(7)};
            transformationDefinedByNode[2] = {node.matrix.at(8), node.matrix.at(9), node.matrix.at(10), node.matrix.at(11)};
            transformationDefinedByNode[3] = {node.matrix.at(12), node.matrix.at(13), node.matrix.at(14), node.matrix.at(15)};
        }

        glm::mat4 invTranMatrix = glm::inverseTranspose(transformationDefinedByNode);
        normalMatrixDefinedByNode[0] = {invTranMatrix[0].x, invTranMatrix[0].y, invTranMatrix[0].z};
        normalMatrixDefinedByNode[1] = {invTranMatrix[1].x, invTranMatrix[1].y, invTranMatrix[1].z};
        normalMatrixDefinedByNode[2] = {invTranMatrix[2].x, invTranMatrix[2].y, invTranMatrix[2].z};

        bool containsNaN = false;
        for(uint32_t i = 0; i < 3; i++) {
            if( std::isnan(normalMatrixDefinedByNode[i].x) ||
                std::isnan(normalMatrixDefinedByNode[i].y) ||
                std::isnan(normalMatrixDefinedByNode[i].z)) {
                containsNaN = true;
                break;
            }
        }
        if(containsNaN) {
            // Usually only happens if the initial transformation matrix is invalid
            normalMatrixDefinedByNode = glm::mat3(1.0);
        }
    }

    // Specification requires this multiplication order
    glm::mat4 transformationMatrix = node.matrix.size() == 16 ? partialTransform * transformationDefinedByNode : partialTransform * translationMatrix * rotationMatrix * scaleMatrix;
    glm::mat3 normalMatrix = node.matrix.size() == 16 ? partialNormalMatrix *  normalMatrixDefinedByNode : partialNormalMatrix * directionMatrix;

    if(node.mesh >= 0 && node.mesh < int(meshTransformationMatrices.size())) {
        meshTransformationMatrices.at(node.mesh) = transformationMatrix;
        normalMatrices.at(node.mesh) = normalMatrix;
    }

    // Call recursively to visit the entire scene graph
    for(const int childNodeIndex : node.children) {
        computeMeshTransformation(nodes, childNodeIndex, transformationMatrix, normalMatrix, meshTransformationMatrices, normalMatrices);
    }
}

ShapeDescriptor::cpu::Mesh ShapeDescriptor::loadGLTFMesh(std::filesystem::path filePath, ShapeDescriptor::RecomputeNormals recomputeNormals) {
    tinygltf::Model model = readTinyGLTFFile(filePath);

    // Calculate transformation matrices
    std::vector<glm::mat4> meshTransformationMatrices(model.meshes.size(), glm::mat4(1));
    std::vector<glm::mat3> meshNormalMatrices(meshTransformationMatrices.size(), glm::mat3(1));
    for(tinygltf::Scene &scene : model.scenes) {
        for(int nodeIndex : scene.nodes) {
            computeMeshTransformation(model.nodes, nodeIndex, glm::mat4(1.0), glm::mat3(1.0), meshTransformationMatrices, meshNormalMatrices);
        }
    }

    size_t vertexCount = 0;
    bool readNormals = true;
    bool readVertexColours = true;

    for(const tinygltf::Mesh& mesh : model.meshes) {
        for(const tinygltf::Primitive& primitive : mesh.primitives) {
            GLTFDrawMode mode = static_cast<GLTFDrawMode>(primitive.mode);

            switch(mode) {
                case GLTFDrawMode::POINTS:
                    if(meshIncludesPointsBehaviour == UnsupportedDrawModeBehaviour::THROW_ERROR) {
                        reportDrawModeError(filePath, primitive.mode);
                    } else {
                        continue;
                    }

                case GLTFDrawMode::LINES:
                case GLTFDrawMode::LINE_LOOP:
                case GLTFDrawMode::LINE_STRIP:
                    if(meshIncludesLinesBehaviour == UnsupportedDrawModeBehaviour::THROW_ERROR) {
                        reportDrawModeError(filePath, primitive.mode);
                    } else {
                        continue;
                    }
                    break;

                case GLTFDrawMode::TRIANGLE_STRIP:
                case GLTFDrawMode::TRIANGLE_FAN:
                    if(meshIncludesTriangleStripOrFanBehaviour == UnsupportedDrawModeBehaviour::THROW_ERROR) {
                        reportDrawModeError(filePath, primitive.mode);
                    } else {
                        continue;
                    }
                    break;
                case GLTFDrawMode::TRIANGLES:
                    // Supported
                    break;
                case GLTFDrawMode::END:
                    break;
            }

            // Reading vertex coordinates
            if(primitive.indices < 0) {
                size_t vertexAccessorID = primitive.attributes.at("POSITION");
                const tinygltf::Accessor& vertexAccessor = model.accessors.at(vertexAccessorID);
                vertexCount += vertexAccessor.count;
            } else {
                vertexCount += model.accessors.at(primitive.indices).count;
            }
            // coordinates are always 3D floats

            // Reading normals
            // These are guaranteed to be 3D floats
            if(!primitive.attributes.contains("NORMAL") && recomputeNormals == RecomputeNormals::DO_NOT_RECOMPUTE) {
                readNormals = false;
            }

            // Reading vertex colours
            if(primitive.attributes.contains("COLOR_0")) {
                int accessorID = primitive.attributes.at("COLOR_0");
                if(model.accessors.at(accessorID).type == TINYGLTF_TYPE_VEC3 || model.accessors.at(accessorID).type == TINYGLTF_TYPE_VEC4) {
                    if(model.accessors.at(accessorID).componentType != TINYGLTF_PARAMETER_TYPE_FLOAT
                    && model.accessors.at(accessorID).componentType != TINYGLTF_PARAMETER_TYPE_UNSIGNED_BYTE
                    && model.accessors.at(accessorID).componentType != TINYGLTF_PARAMETER_TYPE_UNSIGNED_SHORT) {
                        throw std::runtime_error("The file loaded from " + filePath.string() + " specifies vertex colours in a format different from 32-bit floats, 16-bits, or 8 bits for each channel. Please re-export the model to correct this.");
                    }
                } else {
                    throw std::runtime_error("The file loaded from " + filePath.string() +
                                             " specifies vertex colours an invalid number of channels. Please re-export the object using 3 or 4 channel colours.");
                }
            } else {
                readVertexColours = false;
            }
        }
    }



    ShapeDescriptor::cpu::Mesh mesh(vertexCount);

    size_t nextVertexIndex = 0;

    for(uint32_t meshIndex = 0; meshIndex < model.meshes.size(); meshIndex++) {
        const tinygltf::Mesh& modelMesh = model.meshes.at(meshIndex);
        glm::mat4 meshTransformationMatrix = meshTransformationMatrices.at(meshIndex);
        glm::mat3 meshNormalMatrix = meshNormalMatrices.at(meshIndex);

        for (const tinygltf::Primitive &primitive: modelMesh.primitives) {
            GLTFDrawMode mode = static_cast<GLTFDrawMode>(primitive.mode);

            // At this point, either an error was reported, or the unsupported format is supposed to be ignored.
            switch(mode) {
                case GLTFDrawMode::POINTS:
                    continue;
                case GLTFDrawMode::LINES:
                case GLTFDrawMode::LINE_LOOP:
                case GLTFDrawMode::LINE_STRIP:
                    continue;
                case GLTFDrawMode::TRIANGLE_STRIP:
                case GLTFDrawMode::TRIANGLE_FAN:
                    continue;
                case GLTFDrawMode::TRIANGLES:
                case GLTFDrawMode::END:
                    break;
            }

            size_t vertexAccessorID = primitive.attributes.at("POSITION");
            const tinygltf::Accessor& vertexAccessor = model.accessors.at(vertexAccessorID);
            const tinygltf::BufferView& vertexBufferView = model.bufferViews.at(vertexAccessor.bufferView);
            size_t vertexStride = vertexBufferView.byteStride != 0 ? vertexBufferView.byteStride : sizeof(ShapeDescriptor::cpu::float3);
            const unsigned char* vertexBufferBasePointer = model.buffers.at(vertexBufferView.buffer).data.data() + vertexBufferView.byteOffset + vertexAccessor.byteOffset;

            size_t normalAccessorID = 0;
            tinygltf::Accessor* normalAccessor = nullptr;
            tinygltf::BufferView* normalBufferView = nullptr;
            bool primitiveContainsNormals = primitive.attributes.contains("NORMAL");
            bool readNormalsFromThisPrimitive = readNormals && primitiveContainsNormals;
            size_t normalStride = 0;
            unsigned char* normalBufferBasePointer = nullptr;
            if(readNormalsFromThisPrimitive) {
                normalAccessorID = primitive.attributes.at("NORMAL");
                normalAccessor = &model.accessors.at(normalAccessorID);
                normalBufferView = &model.bufferViews.at(normalAccessor->bufferView);
                normalStride = normalBufferView->byteStride != 0 ? normalBufferView->byteStride : sizeof(ShapeDescriptor::cpu::float3);
                normalBufferBasePointer = model.buffers.at(normalBufferView->buffer).data.data() + normalBufferView->byteOffset + normalAccessor->byteOffset;
            }

            size_t colourAccessorID = 0;
            tinygltf::Accessor* colourAccessor = nullptr;
            tinygltf::BufferView* colourBufferView = nullptr;
            size_t colourStride = 0;
            unsigned char* colourBufferBasePointer = nullptr;
            if(readVertexColours) {
                colourAccessorID = primitive.attributes.at("COLOR_0");
                colourAccessor = &model.accessors.at(colourAccessorID);
                colourBufferView = &model.bufferViews.at(colourAccessor->bufferView);

                size_t elementCount = 3;
                size_t bytesPerChannel = sizeof(unsigned char);

                if(colourAccessor->type == TINYGLTF_TYPE_VEC4) {
                    elementCount = 4;
                }
                if(colourAccessor->componentType == TINYGLTF_COMPONENT_TYPE_FLOAT) {
                    bytesPerChannel = sizeof(float);
                }
                if(colourAccessor->componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
                    bytesPerChannel = sizeof(uint16_t);
                }

                colourStride = colourBufferView->byteStride != 0 ? colourBufferView->byteStride : elementCount * bytesPerChannel;
                colourBufferBasePointer = model.buffers.at(colourBufferView->buffer).data.data() + colourBufferView->byteOffset + colourAccessor->byteOffset;
            }


            size_t vertexCountInPrimitive = 0;

            tinygltf::Accessor indexAccessor;
            tinygltf::BufferView indexBufferView;
            unsigned char* indexBasePointer = nullptr;

            bool useIndexBuffer = primitive.indices >= 0;
            if(!useIndexBuffer) {
                size_t vertexAccessorID = primitive.attributes.at("POSITION");
                const tinygltf::Accessor& vertexAccessor = model.accessors.at(vertexAccessorID);
                vertexCountInPrimitive = vertexAccessor.count;
            } else {
                vertexCountInPrimitive = model.accessors.at(primitive.indices).count;
                indexAccessor = model.accessors.at(primitive.indices);
                indexBufferView = model.bufferViews.at(indexAccessor.bufferView);
                indexBasePointer = model.buffers.at(indexBufferView.buffer).data.data() + indexBufferView.byteOffset + indexAccessor.byteOffset;
            }

            for(size_t index = 0; index < vertexCountInPrimitive; index++) {
                size_t vertexIndex = 0;
                if(useIndexBuffer) {
                    unsigned char* indexPointer;
                    switch(indexAccessor.componentType) {
                        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
                            indexPointer = indexBasePointer + index * (indexBufferView.byteStride != 0 ? indexBufferView.byteStride : sizeof(unsigned int));
                            vertexIndex = *reinterpret_cast<unsigned int*>(indexPointer);
                            break;
                        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
                            indexPointer = indexBasePointer + index * (indexBufferView.byteStride != 0 ? indexBufferView.byteStride : sizeof(unsigned short));
                            vertexIndex = *reinterpret_cast<unsigned short*>(indexPointer);
                            break;
                        case TINYGLTF_PARAMETER_TYPE_UNSIGNED_BYTE:
                            indexPointer = indexBasePointer + index * (indexBufferView.byteStride != 0 ? indexBufferView.byteStride : sizeof(unsigned char));
                            vertexIndex = *indexPointer;
                            break;
                        default:
                            throw std::runtime_error("The file loaded from " + filePath.string() +
                                                     " specifies indices with a data type other than unsigned int, unsigned short, or unsigned byte.");
                    }
                } else {
                    vertexIndex = index;
                }

                ShapeDescriptor::cpu::float3 nonTransformedVertex = *reinterpret_cast<const ShapeDescriptor::cpu::float3*>(vertexBufferBasePointer + vertexIndex * vertexStride);
                glm::vec4 transformedVertex = meshTransformationMatrix * glm::vec4(nonTransformedVertex.x, nonTransformedVertex.y, nonTransformedVertex.z, 1.0);
                mesh.vertices[nextVertexIndex] = {transformedVertex.x, transformedVertex.y, transformedVertex.z};

                if(readNormalsFromThisPrimitive) {
                    ShapeDescriptor::cpu::float3 nonTransformedNormal = *reinterpret_cast<ShapeDescriptor::cpu::float3*>(normalBufferBasePointer + vertexIndex * normalStride);
                    glm::vec3 transformedNormal = meshNormalMatrix * glm::vec3(nonTransformedNormal.x, nonTransformedNormal.y, nonTransformedNormal.z);
                    mesh.normals[nextVertexIndex] = {transformedNormal.x, transformedNormal.y, transformedNormal.z};
                }

                if(readVertexColours) {
                    ShapeDescriptor::cpu::uchar4 colour;
                    unsigned char* colourPointer = colourBufferBasePointer + vertexIndex * colourStride;
                    if(colourAccessor->componentType == TINYGLTF_COMPONENT_TYPE_FLOAT) {
                        ShapeDescriptor::cpu::float4 floatColour;
                        if(colourAccessor->type == TINYGLTF_TYPE_VEC3) {
                            ShapeDescriptor::cpu::float3* vec3Pointer = reinterpret_cast<ShapeDescriptor::cpu::float3*>(colourPointer);
                            floatColour = ShapeDescriptor::cpu::float4{vec3Pointer->x, vec3Pointer->y, vec3Pointer->z, 1};
                        } else if(colourAccessor->type == TINYGLTF_TYPE_VEC4) {
                            floatColour = *reinterpret_cast<ShapeDescriptor::cpu::float4*>(colourPointer);
                        } else {
                            throw std::runtime_error("The file loaded from " + filePath.string() +
                                                     " specifies colours with more or fewer than three or four channels.");
                        }
                        colour.r = static_cast<unsigned char>(floatColour.x * 255.0);
                        colour.g = static_cast<unsigned char>(floatColour.y * 255.0);
                        colour.b = static_cast<unsigned char>(floatColour.z * 255.0);
                        colour.a = static_cast<unsigned char>(floatColour.w * 255.0);
                    } else if(colourAccessor->componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
                        if(colourAccessor->type == TINYGLTF_TYPE_VEC3) {
                            colour = {colourPointer[0], colourPointer[1], colourPointer[2], 255};
                        } else if(colourAccessor->type == TINYGLTF_TYPE_VEC4) {
                            colour = *reinterpret_cast<ShapeDescriptor::cpu::uchar4*>(colourPointer);
                        } else {
                            throw std::runtime_error("The file loaded from " + filePath.string() +
                                                     " specifies colours with more or fewer than three or four channels.");
                        }
                    } else if(colourAccessor->componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
                        const uint16_t* colour16Pointer = reinterpret_cast<uint16_t*>(colourPointer);
                        if(colourAccessor->type == TINYGLTF_TYPE_VEC3) {

                            colour = {scale16BitColourTo8Bit(colour16Pointer[0]),
                                      scale16BitColourTo8Bit(colour16Pointer[1]),
                                      scale16BitColourTo8Bit(colour16Pointer[2]), 255};
                        } else if(colourAccessor->type == TINYGLTF_TYPE_VEC4) {
                            colour = {scale16BitColourTo8Bit(colour16Pointer[0]),
                                      scale16BitColourTo8Bit(colour16Pointer[1]),
                                      scale16BitColourTo8Bit(colour16Pointer[2]),
                                      scale16BitColourTo8Bit(colour16Pointer[3])};
                        } else {
                            throw std::runtime_error("The file loaded from " + filePath.string() +
                                                     " specifies colours with more or fewer than three or four channels.");
                        }
                    } else {
                        throw std::runtime_error("The file loaded from " + filePath.string() +
                                                 " specifies colours with a data type that is different from one float or one byte per channel.");
                    }
                    mesh.vertexColours[nextVertexIndex] = colour;
                }

                bool triangleProcessed = nextVertexIndex % 3 == 2;
                bool recomputeNormalBecauseMissing = recomputeNormals == RecomputeNormals::RECOMPUTE_IF_MISSING && primitiveContainsNormals;
                bool recomputeNormalBecauseForced = recomputeNormals == RecomputeNormals::ALWAYS_RECOMPUTE;
                if(triangleProcessed && (recomputeNormalBecauseMissing || recomputeNormalBecauseForced)) {
                    ShapeDescriptor::cpu::float3 normal = computeTriangleNormal(
                            mesh.vertices[nextVertexIndex - 2],
                            mesh.vertices[nextVertexIndex - 1],
                            mesh.vertices[nextVertexIndex]);

                    mesh.normals[nextVertexIndex - 2] = normal;
                    mesh.normals[nextVertexIndex - 1] = normal;
                    mesh.normals[nextVertexIndex] = normal;
                }

                nextVertexIndex++;
            }
        }
    }

    return mesh;
}

ShapeDescriptor::cpu::PointCloud ShapeDescriptor::loadGLTFPointCloud(std::filesystem::path filePath) {
    tinygltf::Model model = readTinyGLTFFile(filePath);

    // Calculate transformation matrices
    std::vector<glm::mat4> meshTransformationMatrices(model.meshes.size(), glm::mat4(1));
    std::vector<glm::mat3> meshNormalMatrices(meshTransformationMatrices.size(), glm::mat3(1));
    for(tinygltf::Scene &scene : model.scenes) {
        for(int nodeIndex : scene.nodes) {
            computeMeshTransformation(model.nodes, nodeIndex, glm::mat4(1.0), glm::mat3(1.0), meshTransformationMatrices, meshNormalMatrices);
        }
    }

    size_t vertexCount = 0;
    bool readNormals = true;
    bool readVertexColours = true;

    for(const tinygltf::Mesh& mesh : model.meshes) {
        for(const tinygltf::Primitive& primitive : mesh.primitives) {
            GLTFDrawMode mode = static_cast<GLTFDrawMode>(primitive.mode);
            // Enforce pure point cloud file
            switch(mode) {
                case GLTFDrawMode::POINTS:
                    break;
                case GLTFDrawMode::LINES:
                case GLTFDrawMode::LINE_LOOP:
                case GLTFDrawMode::LINE_STRIP:
                    if(pointCloudIncludesLinesBehaviour == UnsupportedDrawModeBehaviour::THROW_ERROR) {
                        reportDrawModeError(filePath, primitive.mode);
                    } else {
                        continue;
                    }
                    break;

                case GLTFDrawMode::TRIANGLE_STRIP:
                case GLTFDrawMode::TRIANGLE_FAN:
                    if(pointCloudIncludesTrianglesStripOrFanBehaviour == UnsupportedDrawModeBehaviour::THROW_ERROR) {
                        reportDrawModeError(filePath, primitive.mode);
                    } else {
                        continue;
                    }
                    break;
                case GLTFDrawMode::TRIANGLES:
                    if(pointCloudIncludesTrianglesBehaviour == UnsupportedDrawModeBehaviour::THROW_ERROR) {
                        reportDrawModeError(filePath, primitive.mode);
                    } else {
                        continue;
                    }
                    break;
                case GLTFDrawMode::END:
                    break;
            }

            // Reading vertex coordinates
            if(primitive.indices < 0) {
                size_t vertexAccessorID = primitive.attributes.at("POSITION");
                const tinygltf::Accessor& vertexAccessor = model.accessors.at(vertexAccessorID);
                vertexCount += vertexAccessor.count;
            } else {
                vertexCount += model.accessors.at(primitive.indices).count;
            }

            // coordinates are always 3D floats

            if(primitive.attributes.contains("NORMAL")) {
                readNormals = false;
            }

            // Reading vertex colours
            if(primitive.attributes.contains("COLOR_0")) {
                int accessorID = primitive.attributes.at("COLOR_0");
                if(model.accessors.at(accessorID).type == TINYGLTF_TYPE_VEC3 || model.accessors.at(accessorID).type == TINYGLTF_TYPE_VEC4) {
                    if(model.accessors.at(accessorID).componentType != TINYGLTF_PARAMETER_TYPE_FLOAT
                    && model.accessors.at(accessorID).componentType != TINYGLTF_PARAMETER_TYPE_UNSIGNED_BYTE
                    && model.accessors.at(accessorID).componentType != TINYGLTF_PARAMETER_TYPE_UNSIGNED_SHORT) {
                        throw std::runtime_error("The file loaded from " + filePath.string() + " specifies vertex colours in a format different from 32-bit floats, 16 bits, or 8 bits for each channel. Please re-export the model to correct this.");
                    }
                } else {
                    throw std::runtime_error("The file loaded from " + filePath.string() +
                                             " specifies vertex colours an invalid number of channels. Please re-export the object using 3 or 4 channel colours.");
                }
            } else {
                readVertexColours = false;
            }
        }
    }



    ShapeDescriptor::cpu::PointCloud cloud(vertexCount);

    size_t nextVertexIndex = 0;

    for(const tinygltf::Mesh& modelMesh : model.meshes) {
        for (const tinygltf::Primitive &primitive: modelMesh.primitives) {
            GLTFDrawMode mode = static_cast<GLTFDrawMode>(primitive.mode);

            if(mode != GLTFDrawMode::POINTS) {
                continue;
            }

            size_t vertexAccessorID = primitive.attributes.at("POSITION");
            const tinygltf::Accessor& vertexAccessor = model.accessors.at(vertexAccessorID);
            const tinygltf::BufferView& vertexBufferView = model.bufferViews.at(vertexAccessor.bufferView);
            size_t vertexStride = vertexBufferView.byteStride != 0 ? vertexBufferView.byteStride : sizeof(ShapeDescriptor::cpu::float3);
            const unsigned char* vertexBufferBasePointer = model.buffers.at(vertexBufferView.buffer).data.data() + vertexBufferView.byteOffset + vertexAccessor.byteOffset;

            size_t normalAccessorID = 0;
            tinygltf::Accessor* normalAccessor = nullptr;
            tinygltf::BufferView* normalBufferView = nullptr;
            bool primitiveContainsNormals = primitive.attributes.contains("NORMAL");
            bool readNormalsFromThisPrimitive = readNormals && primitiveContainsNormals;
            size_t normalStride = 0;
            unsigned char* normalBufferBasePointer = nullptr;
            if(readNormalsFromThisPrimitive) {
                normalAccessorID = primitive.attributes.at("NORMAL");
                normalAccessor = &model.accessors.at(normalAccessorID);
                normalBufferView = &model.bufferViews.at(normalAccessor->bufferView);
                normalStride = normalBufferView->byteStride != 0 ? normalBufferView->byteStride : sizeof(ShapeDescriptor::cpu::float3);
                normalBufferBasePointer = model.buffers.at(normalBufferView->buffer).data.data() + normalBufferView->byteOffset + normalAccessor->byteOffset;
            }

            size_t colourAccessorID = 0;
            tinygltf::Accessor* colourAccessor = nullptr;
            tinygltf::BufferView* colourBufferView = nullptr;
            size_t colourStride = 0;
            unsigned char* colourBufferBasePointer = nullptr;
            if(readVertexColours) {
                colourAccessorID = primitive.attributes.at("COLOR_0");
                colourAccessor = &model.accessors.at(colourAccessorID);
                colourBufferView = &model.bufferViews.at(colourAccessor->bufferView);

                size_t elementCount = 3;
                size_t bytesPerChannel = sizeof(unsigned char);

                if(colourAccessor->type == TINYGLTF_TYPE_VEC4) {
                    elementCount = 4;
                }
                if(colourAccessor->componentType == TINYGLTF_COMPONENT_TYPE_FLOAT) {
                    bytesPerChannel = sizeof(float);
                }
                if(colourAccessor->componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
                    bytesPerChannel = sizeof(uint16_t);
                }

                colourStride = colourBufferView->byteStride != 0 ? colourBufferView->byteStride : elementCount * bytesPerChannel;
                colourBufferBasePointer = model.buffers.at(colourBufferView->buffer).data.data() + colourBufferView->byteOffset + colourAccessor->byteOffset;
            }



            size_t vertexCountInPrimitive = 0;

            tinygltf::Accessor indexAccessor;
            tinygltf::BufferView indexBufferView;
            unsigned char* indexBasePointer = nullptr;

            bool useIndexBuffer = primitive.indices >= 0;
            if(!useIndexBuffer) {
                size_t vertexAccessorID = primitive.attributes.at("POSITION");
                const tinygltf::Accessor& vertexAccessor = model.accessors.at(vertexAccessorID);
                vertexCountInPrimitive = vertexAccessor.count;
            } else {
                vertexCountInPrimitive = model.accessors.at(primitive.indices).count;
                indexAccessor = model.accessors.at(primitive.indices);
                indexBufferView = model.bufferViews.at(indexAccessor.bufferView);
                indexBasePointer = model.buffers.at(indexBufferView.buffer).data.data() + indexBufferView.byteOffset + indexAccessor.byteOffset;
            }

            for(size_t index = 0; index < vertexCountInPrimitive; index++) {
                size_t vertexIndex = 0;
                if(useIndexBuffer) {
                    unsigned char* indexPointer;
                    switch(indexAccessor.componentType) {
                        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
                            indexPointer = indexBasePointer + index * (indexBufferView.byteStride != 0 ? indexBufferView.byteStride : sizeof(unsigned int));
                            vertexIndex = *reinterpret_cast<unsigned int*>(indexPointer);
                            break;
                        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
                            indexPointer = indexBasePointer + index * (indexBufferView.byteStride != 0 ? indexBufferView.byteStride : sizeof(unsigned short));
                            vertexIndex = *reinterpret_cast<unsigned short*>(indexPointer);
                            break;
                        case TINYGLTF_PARAMETER_TYPE_UNSIGNED_BYTE:
                            indexPointer = indexBasePointer + index * (indexBufferView.byteStride != 0 ? indexBufferView.byteStride : sizeof(unsigned char));
                            vertexIndex = *indexPointer;
                            break;
                        default:
                            throw std::runtime_error("The file loaded from " + filePath.string() +
                                                     " specifies indices with a data type other than unsigned int, unsigned short, or unsigned byte.");
                    }
                } else {
                    vertexIndex = index;
                }

                cloud.vertices[nextVertexIndex] = *reinterpret_cast<const ShapeDescriptor::cpu::float3*>(vertexBufferBasePointer + vertexIndex * vertexStride);

                if(readNormalsFromThisPrimitive) {
                    cloud.normals[nextVertexIndex] = *reinterpret_cast<ShapeDescriptor::cpu::float3*>(normalBufferBasePointer + vertexIndex * normalStride);
                }

                if(readVertexColours) {
                    ShapeDescriptor::cpu::uchar4 colour;
                    unsigned char* colourPointer = colourBufferBasePointer + vertexIndex * colourStride;
                    if(colourAccessor->componentType == TINYGLTF_COMPONENT_TYPE_FLOAT) {
                        ShapeDescriptor::cpu::float4 floatColour;
                        if(colourAccessor->type == TINYGLTF_TYPE_VEC3) {
                            ShapeDescriptor::cpu::float3* vec3Pointer = reinterpret_cast<ShapeDescriptor::cpu::float3*>(colourPointer);
                            floatColour = ShapeDescriptor::cpu::float4{vec3Pointer->x, vec3Pointer->y, vec3Pointer->z, 1};
                        } else if(colourAccessor->type == TINYGLTF_TYPE_VEC4) {
                            floatColour = *reinterpret_cast<ShapeDescriptor::cpu::float4*>(colourPointer);
                        } else {
                            throw std::runtime_error("The file loaded from " + filePath.string() +
                                                     " specifies colours with more or fewer than three or four channels.");
                        }
                        colour.r = static_cast<unsigned char>(floatColour.x * 255.0);
                        colour.g = static_cast<unsigned char>(floatColour.y * 255.0);
                        colour.b = static_cast<unsigned char>(floatColour.z * 255.0);
                        colour.a = static_cast<unsigned char>(floatColour.w * 255.0);
                    } else if(colourAccessor->componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
                        if(colourAccessor->type == TINYGLTF_TYPE_VEC3) {
                            colour = {colourPointer[0], colourPointer[1], colourPointer[2], 1};
                        } else if(colourAccessor->type == TINYGLTF_TYPE_VEC4) {
                            colour = *reinterpret_cast<ShapeDescriptor::cpu::uchar4*>(colourPointer);
                        } else {
                            throw std::runtime_error("The file loaded from " + filePath.string() +
                                                     " specifies colours with more or fewer than three or four channels.");
                        }
                    } else if(colourAccessor->componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
                        const uint16_t* colour16Pointer = reinterpret_cast<uint16_t*>(colourPointer);
                        if(colourAccessor->type == TINYGLTF_TYPE_VEC3) {

                            colour = {scale16BitColourTo8Bit(colour16Pointer[0]),
                                      scale16BitColourTo8Bit(colour16Pointer[1]),
                                      scale16BitColourTo8Bit(colour16Pointer[2]), 255};
                        } else if(colourAccessor->type == TINYGLTF_TYPE_VEC4) {
                            colour = {scale16BitColourTo8Bit(colour16Pointer[0]),
                                      scale16BitColourTo8Bit(colour16Pointer[1]),
                                      scale16BitColourTo8Bit(colour16Pointer[2]),
                                      scale16BitColourTo8Bit(colour16Pointer[3])};
                        } else {
                            throw std::runtime_error("The file loaded from " + filePath.string() +
                                                     " specifies colours with more or fewer than three or four channels.");
                        }
                    } else {
                        throw std::runtime_error("The file loaded from " + filePath.string() +
                                                 " specifies colours with a data type that is different from one float or one byte per channel.");
                    }
                    cloud.vertexColours[nextVertexIndex] = colour;
                }

                nextVertexIndex++;
            }
        }
    }

    return cloud;
}


