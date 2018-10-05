#pragma once

#include "math.h"
#include <algorithm>
#include <ostream>
#include <iostream>

typedef struct float2 {
	float x;
	float y;

	float2 operator- (float2 other) {
		float2 out;
		out.x = x - other.x;
		out.y = y - other.y;
		return out;
	}

	float2 operator* (float other) {
		float2 out;
		out.x = other * x;
		out.y = other * y;
		return out;
	}

	float2 operator/ (float other) {
	    float2 out;
	    out.x = x / other;
	    out.y = y / other;
	    return out;
	}
} float2;


typedef struct float3 {
	float x;
	float y;
	float z;

	float3 operator- (float3 other) {
		float3 out;
		out.x = x - other.x;
		out.y = y - other.y;
		out.z = z - other.z;
		return out;
	}

	float3 operator+ (float3 other) {
		float3 out;
		out.x = x + other.x;
		out.y = y + other.y;
		out.z = z + other.z;
		return out;
	}

	float3 operator* (float3 other) {
		float3 out;
		out.x = other.x * x;
		out.y = other.y * y;
		out.z = other.z * z;
		return out;
	}

	float3 operator/ (float divisor) {
	    float3 out;
	    out.x = x / divisor;
        out.y = y / divisor;
        out.z = z / divisor;
        return out;
	}

	bool operator== (float3 other) {
		return
			(x == other.x) &&
			(y == other.y) &&
			(z == other.z);
	}





} float3;


std::ostream & operator<<(std::ostream & os, float3 vec);

typedef struct float4 {
	float x;
	float y;
	float z;
	float w;
} float4;

typedef struct int3 {
	int x;
	int y;
	int z;
} int3;

float length(float2 vec);
float length(float3 vec);

inline float2 make_float2(float x, float y) {
    float2 out;
    out.x = x;
    out.y = y;
    return out;
}

inline float3 make_float3(float x, float y, float z) {
    float3 out;
    out.x = x;
    out.y = y;
    out.z = z;
    return out;
}

float2 to_float2(float3 vec);

float2 normalize(float2 in);
float3 normalize(float3 in);

float3 operator* (float3 vec, float other);
float3 operator* (float other, float3 vec);

int clamp(int value, int lower, int upper);