#pragma once

// This must be a function as the header definition may not end up being present
// A function will always be compiled using the same settings as the rest of the library
namespace ShapeDescriptor {
    bool isCUDASupportAvailable();
}