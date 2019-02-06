#pragma once

// Separate ones because C++ templates are mean
// (implementing the template in the header would require moving the kernel in here, thereby opening pandora's box)
void shuffle_prefix_scan_float(float *device_input, float* device_output, int elementCount);
void shuffle_prefix_scan_int(int *device_input, int* device_output, int elementCount);
void shuffle_prefix_scan_uint(unsigned int *device_input, unsigned int* device_output, int elementCount);