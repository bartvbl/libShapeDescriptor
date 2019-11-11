#include "vectorTypes.h"
#include <catch2/catch.hpp>
#include <spinImage/cpu/types/float2.h>
#include <spinImage/cpu/types/float3.h>

TEST_CASE("float vector structs", "[vectors]" ) {

    SECTION("float2 length") {
        SpinImage::cpu::float2 vertex;
        vertex.x = 1;
        vertex.y = 0;
        REQUIRE(length(vertex) == 1);
    }

    SECTION("float3 length") {
        SpinImage::cpu::float3 vertex;
        vertex.x = 1;
        vertex.y = 0;
        vertex.z = 0;
        REQUIRE(length(vertex) == 1);
    }
}