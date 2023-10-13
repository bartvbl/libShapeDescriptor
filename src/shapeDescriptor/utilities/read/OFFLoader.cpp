#include <fast-obj/fast_obj.h>
#include <iostream>
#include <shapeDescriptor/shapeDescriptor.h>

ShapeDescriptor::cpu::Mesh ShapeDescriptor::loadOFF(std::filesystem::path src) {
    // Read file contents into a buffer
    FILE* offFile = fopen(src.c_str(), "r");

    fseek(offFile, 0, SEEK_END);
    size_t size = ftell(offFile);

    char* fileContents = new char[size];
    const char* filePointer = fileContents;

    rewind(offFile);
    size_t bytesRead = fread(fileContents, sizeof(char), size, offFile);
    if(bytesRead != size) {
        throw std::runtime_error("Something went wrong while reading the file. Number of read bytes was "
                                 + std::to_string(bytesRead) + " where " + std::to_string(size) + " was expected.");
    }

    // Read header
    if(filePointer[0] != 'O' || filePointer[1] != 'F' || filePointer[2] != 'F') {
        throw std::runtime_error("Incorrect file header detected when loading:\n" + src.string() +
        "\nAre you sure the file exists and it's an OFF file?");
    }

    fclose(offFile);

    filePointer += 4;

    int vertexCount;
    int faceCount;
    int edgeCount;

    filePointer = parse_int(filePointer, &vertexCount);
    filePointer = skip_whitespace(filePointer);
    filePointer = parse_int(filePointer, &faceCount);
    filePointer = skip_whitespace(filePointer);
    filePointer = parse_int(filePointer, &edgeCount);
    filePointer = skip_whitespace(filePointer);
    filePointer++; // Accounts for newline

    ShapeDescriptor::cpu::float3* raw_vertices = new ShapeDescriptor::cpu::float3[vertexCount];
    ShapeDescriptor::cpu::float3* vertices = new ShapeDescriptor::cpu::float3[3 * faceCount];
    ShapeDescriptor::cpu::float3* normals = new ShapeDescriptor::cpu::float3[3 * faceCount];

    for(int i = 0; i < vertexCount; i++) {
        ShapeDescriptor::cpu::float3 coordinate;

        filePointer = parse_float(filePointer, &coordinate.x);
        filePointer = parse_float(filePointer, &coordinate.y);
        filePointer = parse_float(filePointer, &coordinate.z);
        filePointer = skip_whitespace(filePointer);
        filePointer++; // Accounts for newline

        raw_vertices[i] = coordinate;
    }

    for(int face = 0; face < faceCount; face++) {
        int componentCount = 0;
        int index0 = 0;
        int index1 = 0;
        int index2 = 0;

        filePointer = parse_int(filePointer, &componentCount);
        filePointer = skip_whitespace(filePointer);

        if(componentCount != 3) {
            throw std::runtime_error("This loader only support triangles.");
        }

        filePointer = parse_int(filePointer, &index0);
        filePointer = skip_whitespace(filePointer);
        filePointer = parse_int(filePointer, &index1);
        filePointer = skip_whitespace(filePointer);
        filePointer = parse_int(filePointer, &index2);
        filePointer = skip_whitespace(filePointer);
        filePointer++; // Accounts for newline

        vertices[3 * face + 0] = raw_vertices[index0];
        vertices[3 * face + 1] = raw_vertices[index1];
        vertices[3 * face + 2] = raw_vertices[index2];

        ShapeDescriptor::cpu::float3 normal = computeTriangleNormal(vertices[3 * face + 0], vertices[3 * face + 1], vertices[3 * face + 2]);

        normals[3 * face + 0] = normal;
        normals[3 * face + 1] = normal;
        normals[3 * face + 2] = normal;
    }

    delete[] raw_vertices;
    delete[] fileContents;

    ShapeDescriptor::cpu::Mesh mesh;
    mesh.vertexCount = 3 * faceCount;
    mesh.vertices = vertices;
    mesh.normals = normals;

    return mesh;
}