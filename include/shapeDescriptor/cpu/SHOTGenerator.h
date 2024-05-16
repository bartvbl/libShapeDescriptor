#pragma once

#include <shapeDescriptor/shapeDescriptor.h>

namespace ShapeDescriptor {

    template<typename SHOTDescriptor>
    cpu::array<SHOTDescriptor> generateSHOTDescriptorsMultiRadius(
            cpu::PointCloud cloud,
            cpu::array<OrientedPoint> descriptorOrigins,
            const std::vector<float>& supportRadii,
            SHOTExecutionTimes* executionTimes = nullptr) {
        ShapeDescriptor::cpu::array<SHOTDescriptor> descriptors{};

//        std::vector<ShapeDescriptor::cpu::float3>;

        return descriptors;
    }

    template<typename SHOTDescriptor>
    cpu::array<SHOTDescriptor> generateSHOTDescriptors(
            cpu::PointCloud cloud,
            cpu::array<OrientedPoint> descriptorOrigins,
            float supportRadius,
            SHOTExecutionTimes* executionTimes = nullptr) {
        std::vector<float> radii(descriptorOrigins.length, supportRadius);

        return generateSHOTDescriptorsMultiRadius<SHOTDescriptor>(cloud, descriptorOrigins, radii, executionTimes);
    }
}