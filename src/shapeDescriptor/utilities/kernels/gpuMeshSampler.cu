#include <shapeDescriptor/shapeDescriptor.h>
#include <random>
#include <algorithm>

ShapeDescriptor::cpu::PointCloud ShapeDescriptor::sampleMesh(cpu::Mesh mesh, size_t sampleCount, size_t randomSeed) {
    size_t triangleCount = mesh.vertexCount / 3;

    std::vector<float> cumulativeAreaArray(triangleCount);

    double cumulativeArea = 0;
    for(uint32_t i = 0; i < mesh.vertexCount; i += 3) {
        ShapeDescriptor::cpu::float3 vertex0 = mesh.vertices[i];
        ShapeDescriptor::cpu::float3 vertex1 = mesh.vertices[i + 1];
        ShapeDescriptor::cpu::float3 vertex2 = mesh.vertices[i + 2];

        ShapeDescriptor::cpu::float3 AB = vertex1 - vertex0;
        ShapeDescriptor::cpu::float3 AC = vertex2 - vertex0;

        double area = length(cross(AB, AC)) * 0.5;
        cumulativeArea += area;
        cumulativeAreaArray.at(i/3) = float(area);
    }

    cpu::PointCloud pointCloud(sampleCount);

    std::mt19937_64 randomEngine(randomSeed);
    if(cumulativeArea == 0) {
        // Mesh is a simulated point cloud. Sample random vertices instead
        std::uniform_int_distribution<uint32_t> sampleDistribution(0, mesh.vertexCount);
        for(uint32_t i = 0; i < sampleCount; i++) {
            uint32_t sourceIndex = sampleDistribution(randomEngine);
            pointCloud.vertices[i] = mesh.vertices[sourceIndex];
            if(mesh.normals != nullptr) {
                pointCloud.normals[i] = mesh.normals[sourceIndex];
            } else {
                pointCloud.normals[i] = {0, 0, 1};
            }
        }
    } else {
        // Normal mesh, sample weighted by area
        std::uniform_real_distribution<float> sampleDistribution(0, float(cumulativeArea));
        std::uniform_real_distribution<float> coefficientDistribution(0, 1);

        std::vector<float> samplePoints(sampleCount);
        for(uint32_t i = 0; i < sampleCount; i++) {
            samplePoints.at(i) = sampleDistribution(randomEngine);
        }
        std::sort(samplePoints.begin(), samplePoints.end());

        uint32_t currentTriangleIndex = 0;
        #pragma omp parallel for
        for(uint32_t i = 0; i < sampleCount; i++) {
            float sampleAreaPoint = samplePoints.at(i);
            float nextSampleBorder = cumulativeAreaArray.at(currentTriangleIndex);
            while(nextSampleBorder < sampleAreaPoint) {
                currentTriangleIndex++;
                nextSampleBorder = cumulativeAreaArray.at(currentTriangleIndex);
            }

            float v1 = coefficientDistribution(randomEngine);
            float v2 = coefficientDistribution(randomEngine);

            ShapeDescriptor::cpu::float3 vertex0 = mesh.vertices[3 * currentTriangleIndex + 0];
            ShapeDescriptor::cpu::float3 vertex1 = mesh.vertices[3 * currentTriangleIndex + 1];
            ShapeDescriptor::cpu::float3 vertex2 = mesh.vertices[3 * currentTriangleIndex + 2];

            ShapeDescriptor::cpu::float3 samplePoint =
                    (1 - sqrt(v1)) * vertex0 +
                    (sqrt(v1) * (1 - v2)) * vertex1 +
                    (sqrt(v1) * v2) * vertex2;

            ShapeDescriptor::cpu::float3 normal0 = mesh.normals[3 * currentTriangleIndex + 0];
            ShapeDescriptor::cpu::float3 normal1 = mesh.normals[3 * currentTriangleIndex + 1];
            ShapeDescriptor::cpu::float3 normal2 = mesh.normals[3 * currentTriangleIndex + 2];

            ShapeDescriptor::cpu::float3 sampleNormal =
                    (1 - sqrt(v1)) * normal0 +
                    (sqrt(v1) * (1 - v2)) * normal1 +
                    (sqrt(v1) * v2) * normal2;
            sampleNormal = normalize(sampleNormal);
        }
    }

    return pointCloud;
}
