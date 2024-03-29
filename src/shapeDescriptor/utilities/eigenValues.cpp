#include <shapeDescriptor/shapeDescriptor.h>
#include <glm/glm.hpp>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

std::array<ShapeDescriptor::cpu::float3, 3> ShapeDescriptor::internal::computeEigenVectors(std::array<ShapeDescriptor::cpu::float3, 3> columnMajorMatrix) {
    glm::mat3 matrix = {
            columnMajorMatrix.at(0).x, columnMajorMatrix.at(1).x, columnMajorMatrix.at(2).x,
            columnMajorMatrix.at(0).y, columnMajorMatrix.at(1).y, columnMajorMatrix.at(2).y,
            columnMajorMatrix.at(0).z, columnMajorMatrix.at(1).z, columnMajorMatrix.at(2).z
    };

    Eigen::EigenSolver <Eigen::Matrix3f> eigen_solver;
    Eigen::Matrix3f convertedMatrix;
    for (uint32_t i = 0; i < 3; i++) {
        for (uint32_t j = 0; j < 3; j++) {
            convertedMatrix(j, i) = matrix[i][j];
        }
    }

    eigen_solver.compute(convertedMatrix);

    Eigen::EigenSolver <Eigen::Matrix3f>::EigenvectorsType eigen_vectors;
    Eigen::EigenSolver <Eigen::Matrix3f>::EigenvalueType eigen_values;
    eigen_vectors = eigen_solver.eigenvectors();
    eigen_values = eigen_solver.eigenvalues();

    std::array<float, 3> eigenValues = {
            eigen_values.real()(0),
            eigen_values.real()(1),
            eigen_values.real()(2)
    };

    unsigned int longestIndex = 0;
    unsigned int middleIndex = 1;
    unsigned int shortestIndex = 2;

    if (eigenValues.at(longestIndex) < eigenValues.at(middleIndex))
    {
        std::swap(longestIndex, middleIndex);
    }

    if (eigenValues.at(longestIndex) < eigenValues.at(shortestIndex))
    {
        std::swap(longestIndex, shortestIndex);
    }

    if (eigenValues.at(middleIndex) < eigenValues.at(shortestIndex))
    {
        std::swap(shortestIndex, middleIndex);
    }

    Eigen::Vector3f v1 = eigen_vectors.col(longestIndex).real();
    Eigen::Vector3f v2 = eigen_vectors.col(middleIndex).real();
    Eigen::Vector3f v3 = eigen_vectors.col(shortestIndex).real();

    std::array<ShapeDescriptor::cpu::float3, 3> eigenVectors;

    eigenVectors.at(0) = {v1(0), v1(1), v1(2)};
    eigenVectors.at(1) = {v2(0), v2(1), v2(2)};
    eigenVectors.at(2) = {v3(0), v3(1), v3(2)};

    return eigenVectors;
}