#include <iostream>
#include <shapeDescriptor/cpu/types/float4.h>
#include "GLTFLoader.h"

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
// #define TINYGLTF_NOEXCEPTION // optional. disable exception handling.
#include "tiny_gltf.h"
#include "RecomputeNormals.h"
#include "MeshLoadUtils.h"

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

void reportDrawModeError(const std::filesystem::path &filePath, int drawMode) {
    throw std::runtime_error("The file loaded from " + filePath.string() + " contains geometry with an unsupported drawing mode (" + gltfDrawModes.at(drawMode) + "). Please re-export the object to use triangles exclusively, or use an alternate format.");
}

ShapeDescriptor::cpu::Mesh ShapeDescriptor::utilities::loadGLTFMesh(std::filesystem::path filePath, ShapeDescriptor::RecomputeNormals recomputeNormals) {
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string binaryParseErrorMessage;
    std::string binaryParseWarningMessage;
    std::string asciiParseErrorMessage;
    std::string asciiParseWarningMessage;
    bool binarySuccess = true;
    bool asciiSuccess = true;

    // First try to read the file as a binary file. If that fails, try
    binarySuccess = loader.LoadBinaryFromFile(&model, &binaryParseErrorMessage, &binaryParseWarningMessage, filePath);
    if(!binarySuccess) {
        asciiSuccess = loader.LoadASCIIFromFile(&model, &asciiParseErrorMessage, &asciiParseWarningMessage, filePath);
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
            vertexCount += model.accessors.at(primitive.indices).count;
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
                    && model.accessors.at(accessorID).componentType != TINYGLTF_PARAMETER_TYPE_UNSIGNED_BYTE) {
                        throw std::runtime_error("The file loaded from " + filePath.string() + " specifies vertex colours in a format different from 32-bit floats or unsigned char for each channel. Please re-export the model to correct this.");
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

    for(const tinygltf::Mesh& modelMesh : model.meshes) {
        for (const tinygltf::Primitive &primitive: modelMesh.primitives) {
            GLTFDrawMode mode = static_cast<GLTFDrawMode>(primitive.mode);

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

            size_t indexAccessorID = primitive.indices;
            const tinygltf::Accessor& indexAccessor = model.accessors.at(indexAccessorID);
            const tinygltf::BufferView& indexBufferView = model.bufferViews.at(indexAccessor.bufferView);


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
                colourAccessorID = primitive.attributes.contains("COLOR_0");
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

                colourStride = colourBufferView->byteStride != 0 ? colourBufferView->byteStride : elementCount * bytesPerChannel;
                colourBufferBasePointer = model.buffers.at(colourBufferView->buffer).data.data() + colourBufferView->byteOffset + colourAccessor->byteOffset;
            }



            unsigned char* indexBasePointer = model.buffers.at(indexBufferView.buffer).data.data() + indexBufferView.byteOffset + indexAccessor.byteOffset;
            for(size_t index = 0; index < indexAccessor.count; index++) {
                size_t vertexIndex = 0;
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

                mesh.vertices[nextVertexIndex] = *reinterpret_cast<const ShapeDescriptor::cpu::float3*>(vertexBufferBasePointer + vertexIndex * vertexStride);

                if(readNormalsFromThisPrimitive) {
                    mesh.normals[nextVertexIndex] = *reinterpret_cast<ShapeDescriptor::cpu::float3*>(normalBufferBasePointer + vertexIndex * normalStride);
                }

                if(readVertexColours) {
                    ShapeDescriptor::cpu::uchar4 colour;
                    unsigned char* colourPointer = colourBufferBasePointer + vertexIndex * colourStride;
                    if(colourAccessor->componentType == TINYGLTF_COMPONENT_TYPE_FLOAT) {
                        ShapeDescriptor::cpu::float4 floatColour;
                        if(colourAccessor->type == TINYGLTF_TYPE_VEC3) {
                            ShapeDescriptor::cpu::float3* vec3Pointer = reinterpret_cast<ShapeDescriptor::cpu::float3*>(colourPointer);
                            floatColour = {vec3Pointer->x, vec3Pointer->y, vec3Pointer->z, 1};
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


