#pragma once


VertexDescriptors createClassicDescriptors(Mesh device_mesh, CubePartition device_cubePartition, cudaDeviceProp device_information, OutputImageSettings imageSettings, size_t sampleCount);