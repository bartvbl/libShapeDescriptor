#pragma once 

// Controls the size of the spin images being generated
const unsigned int spinImageWidthPixels = 64;

// Some comparisons in the QSI generation process require a check for equivalence.
// This constant controls the sensitivity of these checks.
#define MAX_EQUIVALENCE_ROUNDING_ERROR 0.0001



#ifdef _WIN32
#define M_PI 3.1415926353
#endif