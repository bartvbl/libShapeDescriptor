#include <catch2/catch.hpp>
#include <shapeDescriptor/shapeDescriptor.h>

TEST_CASE("Ensuring volume computation makes sense") {
    SECTION("Volume computation") {
        const float maxSupportRadius = 5;
        const float minSupportRadius = 0.1;

        float totalVolume = 0;
        for(short layer = 0; layer < SHAPE_CONTEXT_LAYER_COUNT; layer++) {
            for(short slice = 0; slice < SHAPE_CONTEXT_VERTICAL_SLICE_COUNT; slice++) {
                totalVolume += ShapeDescriptor::internal::computeBinVolume(slice, layer, minSupportRadius, maxSupportRadius);
            }
        }
        totalVolume *= float(SHAPE_CONTEXT_HORIZONTAL_SLICE_COUNT);

        const float largeSphereVolume = (4.0f / 3.0f) * float(M_PI) * maxSupportRadius * maxSupportRadius * maxSupportRadius;
        const float smallSphereVolume = (4.0f / 3.0f) * float(M_PI) * minSupportRadius * minSupportRadius * minSupportRadius;
        const float supportVolume = largeSphereVolume - smallSphereVolume;

        float volumeDelta = totalVolume - supportVolume;

        REQUIRE(volumeDelta < 0.0001);
    }
}