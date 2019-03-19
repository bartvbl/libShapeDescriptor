#include "testMain.h"
#include <shapeSearch/cpu/types/float2_cpu.h>

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>




TEST_CASE("float2 length", "Description?" ) {
    float2_cpu vertex;
    vertex.x = 1;
    vertex.y = 0;
    REQUIRE(length(vertex) == 1);
}