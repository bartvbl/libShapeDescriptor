#include <catch2/catch_test_macros.hpp>
#include <shapeDescriptor/shapeDescriptor.h>

TEST_CASE("float vector structs", "[vectors]" ) {

    SECTION("float2 length") {
        ShapeDescriptor::cpu::float2 vertex;
        vertex.x = 1;
        vertex.y = 0;
        REQUIRE(length(vertex) == 1);
    }

    SECTION("float3 length") {
        ShapeDescriptor::cpu::float3 vertex;
        vertex.x = 1;
        vertex.y = 0;
        vertex.z = 0;
        REQUIRE(length(vertex) == 1);
    }
}