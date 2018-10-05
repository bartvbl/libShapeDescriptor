#include "geom.hpp"



/*float3 cross(float3 a, float3 b) {
	float3 out;
	out.x =
}*/

float2 to_float2(float3 vec) {
	return make_float2(vec.x, vec.y);
}

float length(float2 vec) {
	return sqrt(vec.x * vec.x + vec.y * vec.y);
}

float length(float3 vec) {
	return sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
}

float2 normalize(float2 in) {
	float2 out;
	float len = length(in);
	out.x = in.x / len;
	out.y = in.y / len;
	return out;
}

float3 normalize(float3 in) {
	float3 out;
	float len = length(in);
	out.x = in.x / len;
	out.y = in.y / len;
	out.z = in.z / len;
	return out;
}

int clamp(int value, int lower, int upper) {
	return std::max(std::min(value, upper), lower);
}

float3 operator* (float3 vec, float other) {
	float3 out;
	out.x = vec.x * other;
	out.y = vec.y * other;
	out.z = vec.z * other;
	return out;
}

float3 operator*(float other, float3 vec) {
	return operator*(vec, other);
}

std::ostream & operator<<(std::ostream & os, float3 vec) {
    os << "(" << vec.x << ", " << vec.y << ", " << vec.z << ")";
    return os;
}