#include <iostream>
#include "GLTFLoader.h"

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
// #define TINYGLTF_NOEXCEPTION // optional. disable exception handling.
#include "tiny_gltf.h"

ShapeDescriptor::cpu::Mesh ShapeDescriptor::utilities::loadGLTF(std::filesystem::path filePath, bool recomputeNormals) {
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



    return ShapeDescriptor::cpu::Mesh();
}
