#include "PLYLoader.h"
#include <fast-obj/fast_obj.h>
#include <iostream>
#include "MeshLoadUtils.h"

ShapeDescriptor::cpu::Mesh ShapeDescriptor::utilities::loadPLY(std::string src, bool recomputeNormals) {
    // Read file contents into a buffer
    FILE* plyFile = fopen(src.c_str(), "r");

    fseek(plyFile, 0, SEEK_END);
    size_t size = ftell(plyFile);

    char* fileContents = new char[size];
    const char* filePointer = fileContents;

    rewind(plyFile);
    fread(fileContents, sizeof(char), size, plyFile);

    fclose(plyFile);

    // Read header
    if(filePointer[0] != 'p' || filePointer[1] != 'l' || filePointer[2] != 'y') {
        throw std::runtime_error("Incorrect file header detected when loading:\n" + src +
                                 "\nAre you sure the file exists and it's an PLY file?");
    }

    filePointer += 4;

    bool isBinary = false;
    bool isLittleEndian = false;
    bool containsNormals = false;
    int vertexCount = 0;
    int faceCount = 0;

    char endiannessSpecifier = '\0';
    char formatSpecifier = '\0';
    char typeSpecifier = '\0';

    bool lineIsEndHeader = false;
    while(!lineIsEndHeader) {
        switch(*filePointer) {
            case 'f':
                // format
                formatSpecifier = *(filePointer + 7);
                if(formatSpecifier == 'a') {
                    // ascii
                    isBinary = false;
                } else if(formatSpecifier == 'b') {
                    // binary
                    isBinary = true;
                    endiannessSpecifier = *(filePointer + 14);
                    if(endiannessSpecifier == 'l'){
                        // little_endian
                        isLittleEndian = true;
                    } else if(endiannessSpecifier == 'b') {
                        // big_endian
                        isLittleEndian = false;
                    }
                }
                break;
            case 'c':
                // comment
                // No need to do anything with this
                break;
            case 'p':
                // property
                formatSpecifier = *(filePointer + 9);
                if(formatSpecifier == 'f') {
                    // property float
                    typeSpecifier = *(filePointer + 15);
                    if(typeSpecifier == 'n') {
                        typeSpecifier = *(filePointer + 16);
                        if(typeSpecifier == 'x' || typeSpecifier == 'y' || typeSpecifier == 'z') {
                            containsNormals = true;
                        }
                    } else if(typeSpecifier != 'x' && typeSpecifier != 'y' && typeSpecifier != 'z') {
                        throw std::runtime_error("Unknown type specifier ('" + std::to_string(typeSpecifier) + "') encountered when loading PLY file: " + src + ".\nPlease try to re-export it using a different program.");
                    }
                } else if(formatSpecifier == 'l') {
                    // must be property list uchar (int or uint) vertex_index
                    if(*(filePointer + 9) != 'l' || *(filePointer + 14) != 'u' || (*(filePointer + 20) != 'i' && *(filePointer + 20) != 'u')) {
                        throw std::runtime_error("Invalid index list types encountered when loading PLY file: " + src + ".\nPlease try to re-export it using a different program.");
                    }
                } else {
                    throw std::runtime_error("Failed to load the PLY file at: " + src + ".\nReason: the PLY loader only supports float type coordinates, and integer indices. You may need to re-export it with correct settings.");
                }
                break;
            case 'e':
                const char next = *(filePointer + 1);
                if(next == 'n') {
                    // end_header
                    lineIsEndHeader = true;
                } else if(next == 'l') {
                    // element
                    const char specification = *(filePointer + 8);
                    if(specification == 'v') {
                        // vertex
                        filePointer = parse_int(filePointer + 15, &vertexCount);
                    } else if(specification == 'f') {
                        // face
                        filePointer = parse_int(filePointer + 13, &faceCount);
                    }
                }
                break;
        }
        filePointer = skip_line(filePointer);
    }

    ShapeDescriptor::cpu::float3* raw_vertices = new ShapeDescriptor::cpu::float3[vertexCount];
    ShapeDescriptor::cpu::float3* raw_normals = new ShapeDescriptor::cpu::float3[vertexCount];
    ShapeDescriptor::cpu::float3* vertices = new ShapeDescriptor::cpu::float3[3 * faceCount];
    ShapeDescriptor::cpu::float3* normals = new ShapeDescriptor::cpu::float3[3 * faceCount];

    if(isBinary) {
        // Vertices are stored in the same way as the struct, so we can simply copy them.
        const ShapeDescriptor::cpu::float3* vertexArrayStart = reinterpret_cast<const ShapeDescriptor::cpu::float3*>(filePointer);
        std::copy(vertexArrayStart, vertexArrayStart + vertexCount, raw_vertices);
        size_t vertexArraySize = vertexCount * sizeof(ShapeDescriptor::cpu::float3);
        filePointer += vertexArraySize;

        if(containsNormals) {
            throw std::runtime_error("PLY file loaded from " + src + " contains normals, which are not supported by this loader.");
        }

        if(!isLittleEndian) {
            throw std::runtime_error("Failed to load PLY file from " + src + ".\nReason: this loader only supports little endian files");
        }

        // The second step is to interpret the indices
        for(int i = 0; i < faceCount; i++) {
            char indexCount = *filePointer;
            if(indexCount != 3) {
                throw std::runtime_error("An error occurred while loading the file: " + src + ".\nThe PLY file contains non-triangulated faces. Please re-export the file, making sure you enable face triangulation.");
            }
            filePointer++;

            int indexVertex0 = *filePointer;
            int indexVertex1 = *(filePointer + sizeof(int));
            int indexVertex2 = *(filePointer + 2 * sizeof(int));

            filePointer += 3 * sizeof(int);

            vertices[3 * i + 0] = raw_vertices[indexVertex0];
            vertices[3 * i + 1] = raw_vertices[indexVertex1];
            vertices[3 * i + 2] = raw_vertices[indexVertex2];

            ShapeDescriptor::cpu::float3 normal = computeTriangleNormal(vertices[3 * i + 0], vertices[3 * i + 1], vertices[3 * i + 2]);

            normals[3 * i + 0] = normal;
            normals[3 * i + 1] = normal;
            normals[3 * i + 2] = normal;
        }
    } else {
        for(int vertexIndex = 0; vertexIndex < vertexCount; vertexIndex++) {
            ShapeDescriptor::cpu::float3 vertex;
            filePointer = parse_float(filePointer, &vertex.x);
            filePointer = skip_whitespace(filePointer);
            filePointer = parse_float(filePointer, &vertex.y);
            filePointer = skip_whitespace(filePointer);
            filePointer = parse_float(filePointer, &vertex.y);
            filePointer = skip_whitespace(filePointer);

            raw_vertices[vertexIndex] = vertex;

            ShapeDescriptor::cpu::float3 normal;

            if(containsNormals && !recomputeNormals) {
                filePointer = parse_float(filePointer, &normal.x);
                filePointer = skip_whitespace(filePointer);
                filePointer = parse_float(filePointer, &normal.y);
                filePointer = skip_whitespace(filePointer);
                filePointer = parse_float(filePointer, &normal.y);
                filePointer = skip_whitespace(filePointer);

                raw_normals[vertexIndex] = normal;
            }

            if((!containsNormals || recomputeNormals) && vertexIndex % 3 == 2) {
                normal = computeTriangleNormal(
                        raw_vertices[3 * vertexIndex - 2],
                        raw_vertices[3 * vertexIndex - 1],
                        raw_vertices[3 * vertexIndex - 0]);

                raw_normals[3 * vertexIndex - 2] = normal;
                raw_normals[3 * vertexIndex - 1] = normal;
                raw_normals[3 * vertexIndex - 0] = normal;
            }

            filePointer = skip_line(filePointer);
        }

        for(int faceIndex = 0; faceIndex < faceCount; faceIndex++) {
            int indexCount;
            filePointer = parse_int(filePointer, &indexCount);
            if(indexCount != 3) {
                throw std::runtime_error("An error occurred while loading the file: " + src + ".\nThe PLY file contains non-triangulated faces. Please re-export the file, making sure you enable face triangulation.");
            }
            filePointer++;

            int indexVertex0;
            int indexVertex1;
            int indexVertex2;

            filePointer = parse_int(filePointer, &indexVertex0);
            filePointer = skip_whitespace(filePointer);
            filePointer = parse_int(filePointer, &indexVertex1);
            filePointer = skip_whitespace(filePointer);
            filePointer = parse_int(filePointer, &indexVertex2);

            vertices[3 * faceIndex + 0] = raw_vertices[indexVertex0];
            vertices[3 * faceIndex + 1] = raw_vertices[indexVertex1];
            vertices[3 * faceIndex + 2] = raw_vertices[indexVertex2];

            normals[3 * faceIndex + 0] = raw_normals[indexVertex0];
            normals[3 * faceIndex + 1] = raw_normals[indexVertex1];
            normals[3 * faceIndex + 2] = raw_normals[indexVertex2];

            filePointer = skip_line(filePointer);
        }
    }

    delete[] raw_vertices;
    delete[] raw_normals;
    delete[] fileContents;

    ShapeDescriptor::cpu::Mesh mesh;
    mesh.vertexCount = 3 * faceCount;
    mesh.vertices = vertices;
    mesh.normals = normals;

    return mesh;
}