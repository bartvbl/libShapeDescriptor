#include "testMain.h"
#include <shapeSearch/cpu/types/float2_cpu.h>
#include <shapeSearch/cpu/types/float3_cpu.h>

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>




TEST_CASE("float vector structs", "[vectors]" ) {

    SECTION("float2 length") {
        float2_cpu vertex;
        vertex.x = 1;
        vertex.y = 0;
        REQUIRE(length(vertex) == 1);
    }

    SECTION("float3 length") {
        float3_cpu vertex;
        vertex.x = 1;
        vertex.y = 0;
        vertex.z = 0;
        REQUIRE(length(vertex) == 1);
    }
}