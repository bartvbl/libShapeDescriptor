#pragma once

inline ShapeDescriptor::cpu::float3 toFloat3(ShapeDescriptor::cpu::double3 a) {
    return {float(a.x), float(a.y), float(a.z)};
}

inline ShapeDescriptor::cpu::double3 toDouble3(ShapeDescriptor::cpu::float3 a) {
    return {double(a.x), double(a.y), double(a.z)};
}