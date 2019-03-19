#include "correlationComputations.h"
#include <catch2/catch.hpp>
#include <shapeSearch/cpu/types/float3_cpu.h>

TEST_CASE("float3 length", "Description?" ) {
    float3_cpu vertex;
    vertex.x = 1;
    vertex.y = 0;
    vertex.z = 0;
    REQUIRE(length(vertex) == 1);
}