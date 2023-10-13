#include <fast-obj/fast_obj.h>
#include <iostream>
#include <shapeDescriptor/shapeDescriptor.h>


ShapeDescriptor::cpu::Mesh ShapeDescriptor::loadPLY(std::filesystem::path src, RecomputeNormals recomputeNormals) {
    // Read file contents into a buffer
    FILE* plyFile = fopen(src.c_str(), "r");

    fseek(plyFile, 0, SEEK_END);
    size_t size = ftell(plyFile);

    char* fileContents = new char[size];
    const char* filePointer = fileContents;

    rewind(plyFile);
    size_t bytesRead = fread(fileContents, sizeof(char), size, plyFile);
    if(bytesRead != size) {
        throw std::runtime_error("Something went wrong while reading the file. Number of read bytes was "
        + std::to_string(bytesRead) + " where " + std::to_string(size) + " was expected.");
    }

    fclose(plyFile);

    // Read header
    if(filePointer[0] != 'p' || filePointer[1] != 'l' || filePointer[2] != 'y') {
        throw std::runtime_error("Incorrect file header detected when loading:\n" + src.string() +
                                 "\nAre you sure the file exists and it's an PLY file?");
    }

    filePointer += 4;

    bool isBinary = false;
    bool isLittleEndian = false;
    bool containsNormals = false;
    bool containsColours = false;
    bool containsFaces = false;
    bool coordinatesUseDouble = false;
    bool seenFloatCoordinate = false;
    bool seenDoubleCoordinate = false;
    unsigned char componentsPerColour = 3;

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
                if((formatSpecifier == 'f') || (formatSpecifier == 'd')) {
                    // property float or property double
                    coordinatesUseDouble = formatSpecifier == 'd';
                    if((seenDoubleCoordinate && !coordinatesUseDouble) || (seenFloatCoordinate && coordinatesUseDouble)) {
                        throw std::runtime_error("Failed to load the PLY file at: " + src.string() + ".\nReason: The file contains coordinates specified in float and double format. Please re-export the file such that all coordinates use either float or double exclusively.");
                    }

                    if(coordinatesUseDouble) {
                        seenDoubleCoordinate = true;
                    } else {
                        seenFloatCoordinate = true;
                    }

                    typeSpecifier = *(filePointer + 15 + (coordinatesUseDouble ? 1 : 0));
                    if(typeSpecifier == 'n') {
                        typeSpecifier = *(filePointer + 16);
                        if(typeSpecifier == 'x' || typeSpecifier == 'y' || typeSpecifier == 'z') {
                            containsNormals = true;
                        }
                    } else if(typeSpecifier != 'x' && typeSpecifier != 'y' && typeSpecifier != 'z') {
                        throw std::runtime_error("Unknown type specifier ('" + std::to_string(typeSpecifier) + "') encountered when loading PLY file: " + src.string() + ".\nPlease try to re-export it using a different program.");
                    }
                } else if(formatSpecifier == 'l') {
                    // must be property list uchar (int or uint) vertex_index
                    if(*(filePointer + 9) != 'l' || *(filePointer + 14) != 'u' || (*(filePointer + 20) != 'i' && *(filePointer + 20) != 'u')) {
                        throw std::runtime_error("Invalid index list types encountered when loading PLY file: " + src.string() + ".\nPlease try to re-export it using a different program.");
                    }
                } else if(formatSpecifier == 'u') {
                    // property uchar
                    typeSpecifier = *(filePointer + 15);
                    if(typeSpecifier == 'r' || typeSpecifier == 'g' || typeSpecifier == 'b' || typeSpecifier == 'a') {
                        if(typeSpecifier == 'a') {
                            componentsPerColour = 4;
                        }
                        containsColours = true;
                    } else {
                        throw std::runtime_error("Failed to load the PLY file at: " + src.string() + ".\nReason: Additional properties detected that are not supported by the PLY loader. You can try re-exporting the file, or use the OBJ format instead.");
                    }
                } else {
                    throw std::runtime_error("Failed to load the PLY file at: " + src.string() + ".\nReason: the PLY loader only supports float type coordinates, and integer indices. You may need to re-export it with correct settings.");
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
                        // Point clouds do not contain faces, thus the feature is optional
                        containsFaces = true;
                        filePointer = parse_int(filePointer + 13, &faceCount);
                    }
                }
                break;
        }
        filePointer = skip_line(filePointer);
    }

    if(coordinatesUseDouble) {
        std::cout << "Warning: PLY file contains coordinates in double format. These will be converted to single precision (float)." << std::endl;
    }

    size_t vertexBufferLength = containsFaces ? 3 * faceCount : vertexCount;

    ShapeDescriptor::cpu::float3* raw_vertices = nullptr;
    ShapeDescriptor::cpu::float3* vertices = nullptr;
    raw_vertices = new ShapeDescriptor::cpu::float3[vertexCount];
    vertices = new ShapeDescriptor::cpu::float3[vertexBufferLength];

    // When a mesh has faces, we can compute normals automatically
    ShapeDescriptor::cpu::float3* raw_normals = nullptr;
    ShapeDescriptor::cpu::float3* normals = nullptr;
    if(containsNormals || containsFaces) {
        raw_normals = new ShapeDescriptor::cpu::float3[vertexCount];
        normals = new ShapeDescriptor::cpu::float3[vertexBufferLength];
    }
    bool shouldRecomputeNormals = (!containsNormals && recomputeNormals == RecomputeNormals::RECOMPUTE_IF_MISSING) || recomputeNormals == RecomputeNormals::ALWAYS_RECOMPUTE;

    ShapeDescriptor::cpu::uchar4* raw_colours = nullptr;
    ShapeDescriptor::cpu::uchar4* colours = nullptr;
    if(containsColours) {
        raw_colours = new ShapeDescriptor::cpu::uchar4[vertexCount];
        colours = new ShapeDescriptor::cpu::uchar4[vertexCount];
    }

    if(isBinary) {
        if(!isLittleEndian) {
            throw std::runtime_error("Failed to load PLY file from " + src.string() + ".\nReason: this loader only supports little endian files");
        }

        if(!containsNormals && !containsColours && !coordinatesUseDouble) {
            // When there are no other properties, vertices are stored in the same way as the struct, so we can simply copy them.
            // This is a shortcut for much faster load times
            const ShapeDescriptor::cpu::float3* vertexArrayStart= reinterpret_cast<const ShapeDescriptor::cpu::float3*>(filePointer);
            std::copy(vertexArrayStart, vertexArrayStart + vertexCount, raw_vertices);
            size_t vertexArraySize = vertexCount * sizeof(ShapeDescriptor::cpu::float3);
            filePointer += vertexArraySize;
        } else {
            // If there are other properties, we need to separate them
            const size_t strideBytes =
                    (coordinatesUseDouble ? sizeof(ShapeDescriptor::cpu::double3) : sizeof(ShapeDescriptor::cpu::float3))
                  + (containsNormals ? sizeof(ShapeDescriptor::cpu::float3) : 0)
                  + (containsColours ? sizeof(ShapeDescriptor::cpu::uchar4) : 0);
            for(size_t i = 0; i < vertexCount; i++) {
                // Copy vertex
                const char* rawVertexPointer = filePointer + (strideBytes * i);
                if(coordinatesUseDouble) {
                    const ShapeDescriptor::cpu::double3* vertexPointer = reinterpret_cast<const ShapeDescriptor::cpu::double3*>(rawVertexPointer);
                    ShapeDescriptor::cpu::float3* destination = containsFaces ? &raw_vertices[i] : &vertices[i];
                    *destination = (*vertexPointer).as_float3();
                } else {
                    const ShapeDescriptor::cpu::float3* vertexPointer = reinterpret_cast<const ShapeDescriptor::cpu::float3*>(rawVertexPointer);
                    ShapeDescriptor::cpu::float3* destination = containsFaces ? &raw_vertices[i] : &vertices[i];
                    *destination = *vertexPointer;
                }

                // Copy normal
                const char* rawNormalPointer = rawVertexPointer + (coordinatesUseDouble ? sizeof(ShapeDescriptor::cpu::double3) : sizeof(ShapeDescriptor::cpu::float3));
                if(containsNormals) {
                    const ShapeDescriptor::cpu::float3* normalPointer = reinterpret_cast<const ShapeDescriptor::cpu::float3*>(rawNormalPointer);
                    ShapeDescriptor::cpu::float3* destination = containsFaces ? &raw_normals[i] : &normals[i];
                    *destination = *normalPointer;
                }

                // Copy colour
                if(containsColours) {
                    ShapeDescriptor::cpu::uchar4* destination = containsFaces ? &raw_colours[i] : &colours[i];
                    const char *rawColourPointer =
                            rawNormalPointer + (containsNormals ? sizeof(ShapeDescriptor::cpu::float3) : 0);
                    if (componentsPerColour == 3) {
                        const ShapeDescriptor::cpu::uchar3 *colourPointer = reinterpret_cast<const ShapeDescriptor::cpu::uchar3 *>(rawColourPointer);
                        ShapeDescriptor::cpu::uchar3 colour = *colourPointer;
                        *destination = {colour.r, colour.g, colour.b, 255};
                    } else {
                        const ShapeDescriptor::cpu::uchar4 *colourPointer = reinterpret_cast<const ShapeDescriptor::cpu::uchar4 *>(rawColourPointer);
                        *destination = *colourPointer;
                    }
                }
            }
        }



        if(containsFaces) {
            // The second step is to interpret the indices
            for (int i = 0; i < faceCount; i++) {
                char indexCount = *filePointer;
                if (indexCount != 3) {
                    throw std::runtime_error("An error occurred while loading the file: " + src.string() +
                                             ".\nThe PLY file contains non-triangulated faces. Please re-export the file, making sure you enable face triangulation.");
                }
                filePointer++;

                int indexVertex0 = *reinterpret_cast<const int *>(filePointer);
                int indexVertex1 = *reinterpret_cast<const int *>(filePointer + sizeof(int));
                int indexVertex2 = *reinterpret_cast<const int *>(filePointer + 2 * sizeof(int));

                filePointer += 3 * sizeof(int);

                vertices[3 * i + 0] = raw_vertices[indexVertex0];
                vertices[3 * i + 1] = raw_vertices[indexVertex1];
                vertices[3 * i + 2] = raw_vertices[indexVertex2];

                if (containsColours) {
                    colours[3 * i + 0] = raw_colours[indexVertex0];
                    colours[3 * i + 1] = raw_colours[indexVertex1];
                    colours[3 * i + 2] = raw_colours[indexVertex2];
                }

                if (shouldRecomputeNormals) {
                    ShapeDescriptor::cpu::float3 normal = computeTriangleNormal(vertices[3 * i + 0],
                                                                                vertices[3 * i + 1],
                                                                                vertices[3 * i + 2]);

                    normals[3 * i + 0] = normal;
                    normals[3 * i + 1] = normal;
                    normals[3 * i + 2] = normal;
                } else {
                    normals[3 * i + 0] = raw_normals[indexVertex0];
                    normals[3 * i + 1] = raw_normals[indexVertex1];
                    normals[3 * i + 2] = raw_normals[indexVertex2];
                }
            }
        }
    } else {
        for(int vertexIndex = 0; vertexIndex < vertexCount; vertexIndex++) {
            ShapeDescriptor::cpu::float3 vertex;
            filePointer = parse_float(filePointer, &vertex.x);
            filePointer = skip_whitespace(filePointer);
            filePointer = parse_float(filePointer, &vertex.y);
            filePointer = skip_whitespace(filePointer);
            filePointer = parse_float(filePointer, &vertex.z);
            filePointer = skip_whitespace(filePointer);

            ShapeDescriptor::cpu::float3* vertexDestination = containsFaces ? &raw_vertices[vertexIndex] : &vertices[vertexIndex];
            *vertexDestination = vertex;

            // Need to advance pointer in case any colours follow, just in case there are colour definitions that follow
            ShapeDescriptor::cpu::float3 normal;
            if(containsNormals || containsColours) {
                filePointer = parse_float(filePointer, &normal.x);
                filePointer = skip_whitespace(filePointer);
                filePointer = parse_float(filePointer, &normal.y);
                filePointer = skip_whitespace(filePointer);
                filePointer = parse_float(filePointer, &normal.z);
                filePointer = skip_whitespace(filePointer);

                if(containsNormals) {
                    ShapeDescriptor::cpu::float3* normalDestination = containsFaces ? &raw_normals[vertexIndex] : &normals[vertexIndex];
                    *normalDestination = normal;
                }
            }

            // Recompute normals if needed or desired, and when vertex information is available
            if(containsFaces && shouldRecomputeNormals && vertexIndex % 3 == 2) {
                normal = computeTriangleNormal(
                        raw_vertices[3 * vertexIndex - 2],
                        raw_vertices[3 * vertexIndex - 1],
                        raw_vertices[3 * vertexIndex - 0]);

                raw_normals[3 * vertexIndex - 2] = normal;
                raw_normals[3 * vertexIndex - 1] = normal;
                raw_normals[3 * vertexIndex - 0] = normal;
            }

            if(containsColours) {
                ShapeDescriptor::cpu::uchar4 colour;
                filePointer = parse_uchar(filePointer, &colour.r);
                filePointer = skip_whitespace(filePointer);
                filePointer = parse_uchar(filePointer, &colour.g);
                filePointer = skip_whitespace(filePointer);
                filePointer = parse_uchar(filePointer, &colour.b);
                filePointer = skip_whitespace(filePointer);
                if(componentsPerColour == 4) {
                    filePointer = parse_uchar(filePointer, &colour.a);
                    filePointer = skip_whitespace(filePointer);
                } else {
                    colour.a = 255;
                }

                ShapeDescriptor::cpu::uchar4* destination = containsFaces ? &raw_colours[vertexIndex] : &colours[vertexIndex];
                *destination = colour;
            }

            filePointer = skip_line(filePointer);
        }

        if(containsFaces) {
            for (int faceIndex = 0; faceIndex < faceCount; faceIndex++) {
                int indexCount;
                filePointer = parse_int(filePointer, &indexCount);
                if (indexCount != 3) {
                    throw std::runtime_error("An error occurred while loading the file: " + src.string() +
                                             ".\nThe PLY file contains non-triangulated faces. Please re-export the file, making sure you enable face triangulation.");
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
    }

    delete[] raw_vertices;
    if(containsNormals) {
        delete[] raw_normals;
    }
    if(containsColours) {
        delete[] raw_colours;
    }
    delete[] fileContents;

    ShapeDescriptor::cpu::Mesh mesh;
    mesh.vertexCount = vertexBufferLength;
    mesh.vertices = vertices;
    mesh.normals = normals;
    mesh.vertexColours = colours;

    return mesh;
}