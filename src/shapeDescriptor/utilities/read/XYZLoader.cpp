#include <fast_obj.h>
#include <vector>
#include <shapeDescriptor/shapeDescriptor.h>

ShapeDescriptor::cpu::PointCloud ShapeDescriptor::loadXYZ(std::filesystem::path src, bool readNormals, bool readColours) {
    // Read file contents into a buffer
    FILE* xyzFile = fopen(src.c_str(), "r");

    fseek(xyzFile, 0, SEEK_END);
    size_t size = ftell(xyzFile);

    char* fileContents = new char[size];
    const char* filePointer = fileContents;

    rewind(xyzFile);
    size_t bytesRead = fread(fileContents, sizeof(char), size, xyzFile);
    if(bytesRead != size) {
        throw std::runtime_error("Something went wrong while reading the file. Number of read bytes was "
                                 + std::to_string(bytesRead) + " where " + std::to_string(size) + " was expected.");
    }

    fclose(xyzFile);

    std::vector<ShapeDescriptor::cpu::float3> vertices;
    std::vector<ShapeDescriptor::cpu::float3> normals;
    std::vector<ShapeDescriptor::cpu::uchar4> colours;

    // Avoids some reallocations
    vertices.reserve(1000);
    normals.reserve(1000);
    colours.reserve(1000);

    while(filePointer < fileContents + size) {
        // If line does not contain any useful data, we skip it
        if(*filePointer == '#' || *filePointer == '\n' || is_whitespace(*filePointer)) {
            filePointer = skip_line(filePointer);
        }

        ShapeDescriptor::cpu::float3 coordinate;

        filePointer = parse_float(filePointer, &coordinate.x);
        filePointer = parse_float(filePointer, &coordinate.y);
        filePointer = parse_float(filePointer, &coordinate.z);
        filePointer = skip_whitespace(filePointer);

        vertices.emplace_back(coordinate);

        if(readNormals) {
            ShapeDescriptor::cpu::float3 normal;

            filePointer = parse_float(filePointer, &normal.x);
            filePointer = parse_float(filePointer, &normal.y);
            filePointer = parse_float(filePointer, &normal.z);
            filePointer = skip_whitespace(filePointer);

            normals.emplace_back(normal);
        }

        if(readColours) {
            ShapeDescriptor::cpu::float3 floatColour;

            filePointer = parse_float(filePointer, &floatColour.x);
            filePointer = parse_float(filePointer, &floatColour.y);
            filePointer = parse_float(filePointer, &floatColour.z);
            filePointer = skip_whitespace(filePointer);

            ShapeDescriptor::cpu::uchar4 colour;
            colour.r = (unsigned char) ((floatColour.x) * 255.0);
            colour.g = (unsigned char) ((floatColour.y) * 255.0);
            colour.b = (unsigned char) ((floatColour.z) * 255.0);
            colour.a = 255;
            colours.emplace_back(colour);
        }

        filePointer++; // Accounts for newline
    }

    ShapeDescriptor::cpu::PointCloud cloud;

    size_t vertexCount = vertices.size();

    cloud.vertices = new ShapeDescriptor::cpu::float3[vertexCount];
    std::copy(vertices.begin(), vertices.end(), cloud.vertices);

    if(readNormals) {
        cloud.normals = new ShapeDescriptor::cpu::float3[vertexCount];
        cloud.hasVertexNormals = true;
        std::copy(normals.begin(), normals.end(), cloud.normals);
    }

    if(readColours) {
        cloud.vertexColours = new ShapeDescriptor::cpu::uchar4[vertexCount];
        cloud.hasVertexColours = true;
        std::copy(colours.begin(), colours.end(), cloud.vertexColours);
    }

    return cloud;
}
