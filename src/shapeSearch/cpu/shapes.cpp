#include "shapes.h"

//int getMiddlePoint(unsigned int p1, unsigned int p2)
//{
	/*// first check if we have it already
	bool firstIsSmaller = p1 < p2;
	unsigned int smallerIndex = firstIsSmaller ? p1 : p2;
	unsigned int greaterIndex = firstIsSmaller ? p2 : p1;
	unsigned int key = (smallerIndex << 32) + greaterIndex;

	// not in cache, calculate it
	float3 point1 = this.geometry.Positions[p1];
	float3 point2 = this.geometry.Positions[p2];
	float3 middle = new float3(
			(point1.X + point2.X) / 2.0,
			(point1.Y + point2.Y) / 2.0,
			(point1.Z + point2.Z) / 2.0);

	// add vertex makes sure point is on unit sphere
	int i = addVertex(middle);

	return i;*/
//}

//Mesh generateSphere(unsigned int recursionLevel) {

	/*const unsigned int baseLevelTriangleCount = 20;
	const unsigned int baseLevelIndexCount = 60;
	const unsigned int baseLevelVertexCount = 12;

	// The base triangle count for no recursion is 20 triangles.
	// Each level multiplies the number of triangles by 4.
	// This can be done using a single bit shift.
	unsigned int triangleCount = baseLevelTriangleCount << (2 * recursionLevel);
	unsigned int indexCount = 3 * triangleCount;

	Mesh outputMesh(triangleCount * 3, indexCount, VERTICES);

	unsigned int index = 0;

	// create 12 vertices of a icosahedron
	float t = (1.0f + float(std::sqrt(5.0))) / 2.0f;


	float3 baseLevelVertices[] = {
		normalize(make_float3(-1, t, 0)),
		normalize(make_float3( 1, t, 0)),
		normalize(make_float3(-1, -t, 0)),
		normalize(make_float3( 1, -t, 0)),

		normalize(make_float3( 0, -1, t)),
		normalize(make_float3( 0, 1, t)),
		normalize(make_float3( 0, -1, -t)),
		normalize(make_float3( 0, 1, -t)),

		normalize(make_float3( t, 0, -1)),
		normalize(make_float3( t, 0, 1)),
		normalize(make_float3(-t, 0, -1)),
		normalize(make_float3(-t, 0, 1))
	};

	unsigned int baseLevelIndices[] = {
		// 5 faces around point 0
		0, 11, 5,
		0, 5, 1,
		0, 1, 7,
		0, 7, 10,
		0, 10, 11,

		// 5 adjacent faces
		1, 5, 9,
		5, 11, 4,
		11, 10, 2,
		10, 7, 6,
		7, 1, 8,

		// 5 faces around point 3
		3, 9, 4,
		3, 4, 2,
		3, 2, 6,
		3, 6, 8,
		3, 8, 9,

		// 5 adjacent faces
		4, 9, 5,
		2, 4, 11,
		6, 2, 10,
		8, 6, 7,
		9, 8, 1
	};

	// Copy local arrays into created Mesh
	float3* vertexStartPointer = outputMesh.vertices + outputMesh.vertexCount - baseLevelVertexCount - 1;
	unsigned int* indexStartPointer = outputMesh.indices + outputMesh.indexCount - baseLevelIndexCount - 1;


	std::copy(&baseLevelVertices, (&baseLevelVertices) + baseLevelVertexCount, outputMesh.vertices;
	std::copy(&baseLevelIndices, (&baseLevelIndices) + baseLevelIndexCount, outputMesh.indices);

	// Subdivide triangles
	for (int i = 0; i < recursionLevel; i++)
	{
		foreach (var tri in faces)
		{
			// replace triangle by 4 triangles
			int a = getMiddlePoint(tri.v1, tri.v2);
			int b = getMiddlePoint(tri.v2, tri.v3);
			int c = getMiddlePoint(tri.v3, tri.v1);

			faces2.Add(new TriangleIndices(tri.v1, a, c));
			faces2.Add(new TriangleIndices(tri.v2, b, a));
			faces2.Add(new TriangleIndices(tri.v3, c, b));
			faces2.Add(new TriangleIndices(a, b, c));
		}
		faces = faces2;
	}

			// done, now add triangles to mesh
			foreach (var tri in faces)
			{
				this.geometry.TriangleIndices.Add(tri.v1);
				this.geometry.TriangleIndices.Add(tri.v2);
				this.geometry.TriangleIndices.Add(tri.v3);
			}

			return this.geometry;
		}
	}*/
//}

/*Mesh generateCylinder(float3 origin, float3 direction, float radius, float height, unsigned int numSlices) {
	float halfHeight = height / 2.0f;

	float3 topDiscOrigin = origin + (direction * halfHeight);
	float3 bottomDiscOrigin = origin - (direction * halfHeight);

	float3 topDiscNormal = direction;
	float3 bottomDiscNormal = direction * -1.0f;

}*/
