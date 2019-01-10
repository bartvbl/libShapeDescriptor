#pragma once

#include "math.h"
#include <algorithm>
#include <ostream>
#include <iostream>

typedef struct float2_cpu {
	float x;
	float y;

	float2_cpu operator- (float2_cpu other) {
		float2_cpu out;
		out.x = x - other.x;
		out.y = y - other.y;
		return out;
	}

	float2_cpu operator* (float other) {
		float2_cpu out;
		out.x = other * x;
		out.y = other * y;
		return out;
	}

	float2_cpu operator/ (float other) {
	    float2_cpu out;
	    out.x = x / other;
	    out.y = y / other;
	    return out;
	}
} float2_cpu;


typedef struct float3_cpu {
	float x;
	float y;
	float z;

	float3_cpu operator- (float3_cpu other) {
		float3_cpu out;
		out.x = x - other.x;
		out.y = y - other.y;
		out.z = z - other.z;
		return out;
	}

	float3_cpu operator+ (float3_cpu other) {
		float3_cpu out;
		out.x = x + other.x;
		out.y = y + other.y;
		out.z = z + other.z;
		return out;
	}

	float3_cpu operator* (float3_cpu other) {
		float3_cpu out;
		out.x = other.x * x;
		out.y = other.y * y;
		out.z = other.z * z;
		return out;
	}

	float3_cpu operator/ (float divisor) {
	    float3_cpu out;
	    out.x = x / divisor;
        out.y = y / divisor;
        out.z = z / divisor;
        return out;
	}

	bool operator== (float3_cpu other) {
		return
			(x == other.x) &&
			(y == other.y) &&
			(z == other.z);
	}
} float3_cpu;


std::ostream & operator<<(std::ostream & os, float3_cpu vec);

typedef struct float4_cpu {
	float x;
	float y;
	float z;
	float w;
} float4_cpu;

typedef struct int3_cpu {
	int x;
	int y;
	int z;
} int3_cpu;

float length(float2_cpu vec);
float length(float3_cpu vec);

inline float2_cpu make_float2_cpu(float x, float y) {
    float2_cpu out;
    out.x = x;
    out.y = y;
    return out;
}

inline float3_cpu make_float3_cpu(float x, float y, float z) {
    float3_cpu out;
    out.x = x;
    out.y = y;
    out.z = z;
    return out;
}

float2_cpu to_float2_cpu(float3_cpu vec);

float2_cpu normalize(float2_cpu in);
float3_cpu normalize(float3_cpu in);

float3_cpu operator* (float3_cpu vec, float other);
float3_cpu operator* (float other, float3_cpu vec);

int clamp(int value, int lower, int upper);