#include <iostream>
#include "GLTFLoader.h"

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
// #define TINYGLTF_NOEXCEPTION // optional. disable exception handling.
#include "tiny_gltf.h"
#include "RecomputeNormals.h"

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

    const std::array<std::string, 7> gltfDrawModes = {"POINTS", "LINES", "LINE_LOOP", "LINE_STRIP", "TRIANGLES", "TRIANGLE_STRIP", "TRIANGLE_FAN"};

    size_t vertexCount = 0;
    bool readNormals = true;
    bool readTextureCoordinates = true;

    for(const tinygltf::Mesh& mesh : model.meshes) {
        for(const tinygltf::Primitive& primitive : mesh.primitives) {
            if(primitive.mode != static_cast<int>(GLTFDrawMode::TRIANGLES)) {
                throw std::runtime_error("The file loaded from " + filePath.string() + " contains geometry with an unsupported drawing mode (" + gltfDrawModes.at(primitive.mode) + "). "
                    "Please re-export the object to use triangles exclusively, or use an alternate format.");
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
            }
        }
    }



    ShapeDescriptor::cpu::Mesh mesh(vertexCount);

    size_t nextVertexIndex = 0;

    for(const tinygltf::Mesh& modelMesh : model.meshes) {
        for (const tinygltf::Primitive &primitive: modelMesh.primitives) {
            size_t indexAccessorID = primitive.indices;
            const tinygltf::Accessor& indexAccessor = model.accessors.at(indexAccessorID);
            const tinygltf::BufferView& indexBufferView = model.bufferViews.at(indexAccessor.bufferView);

            size_t vertexAccessorID = primitive.attributes.at("POSITION");
            const tinygltf::Accessor& vertexAccessor = model.accessors.at(vertexAccessorID);
            const tinygltf::BufferView& vertexBufferView = model.bufferViews.at(vertexAccessor.bufferView);

            unsigned char* indexBasePointer = model.buffers.at(indexBufferView.buffer).data.data() + indexBufferView.byteOffset;
            for(size_t index = 0; index < indexAccessor.count; index++) {
                size_t vertexIndex = 0;
                unsigned char* indexPointer = indexBasePointer + index * indexBufferView.byteStride;
                switch(indexAccessor.componentType) {
                    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
                        vertexIndex = *reinterpret_cast<unsigned int*>(indexPointer);
                        break;
                    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
                        vertexIndex = *reinterpret_cast<unsigned short*>(indexPointer);
                        break;
                    case TINYGLTF_PARAMETER_TYPE_UNSIGNED_BYTE:
                        vertexIndex = *indexPointer;
                        break;
                    default:
                        throw std::runtime_error("The file loaded from " + filePath.string() +
                                                 " specifies indices with a data type other than unsigned int, unsigned short, or unsigned byte.");
                }

                mesh.vertices[nextVertexIndex] = *reinterpret_cast<ShapeDescriptor::cpu::float3*>(model.buffers.at(vertexBufferView.buffer).data.data() + vertexIndex * vertexBufferView.byteStride + vertexBufferView.byteOffset);
                nextVertexIndex++;
            }
        }
    }

    return mesh;
}
