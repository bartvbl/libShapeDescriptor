#include <fast_obj.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <shapeDescriptor/shapeDescriptor.h>

void ShapeDescriptor::writeXYZ(std::filesystem::path destination, ShapeDescriptor::cpu::PointCloud pointCloud) {
    std::string outputFileExtension = destination.extension().string();
    bool exportNormals = outputFileExtension == ".xyzn" || outputFileExtension == ".XYZN";
    bool exportVertexColours = outputFileExtension == ".xyzrgb" || outputFileExtension == ".XYZRGB";

    std::stringstream outputStream;

    outputStream << "# Point cloud contains " << pointCloud.pointCount << " points.\n";

    for(size_t pointIndex = 0; pointIndex < pointCloud.pointCount; pointIndex++) {
        ShapeDescriptor::cpu::float3 vertex = pointCloud.vertices[pointIndex];
        outputStream << vertex.x << " " << vertex.y << " " << vertex.z;

        if(exportNormals && pointCloud.hasVertexNormals) {
            ShapeDescriptor::cpu::float3 normal = pointCloud.normals[pointIndex];
            outputStream << " " << normal.x << " " << normal.y << " " << normal.z;
        }

        if(exportVertexColours && pointCloud.hasVertexColours) {
            ShapeDescriptor::cpu::uchar4 vertexColour = pointCloud.vertexColours[pointIndex];
            outputStream << " " << float(vertexColour.r)/255.0f
                         << " " << float(vertexColour.g)/255.0f
                         << " " << float(vertexColour.b)/255.0f
                         << " " << float(vertexColour.a)/255.0f;
        }

        outputStream << '\n';
    }

    std::ofstream fileStream{destination};
    fileStream << outputStream.str();
}
