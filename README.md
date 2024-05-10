# libShapeDescriptor

This library contains efficient GPU implementations for computing and comparing the following descriptors efficiently on the GPU:

- Spin Image (Johnson et al., 1999)
- 3D Shape Context (Frome et al., 2004)
- Fast Point Feature Histogram (Rusu et al., 2009)
- Unique Shape Context (Tombari et al., 2010)
- Signature of Histograms of OrienTations (Salti et al., 2014)
- Radial Intersection Count Image (van Blokland et al., 2020)
- Quick Intersection Count Change Image (van Blokland et al., 2020)

Along with a set of useful utility functions surrounding these.

The library also contains some search related implementations, but they were more aimed at satisfying the needs for our Clutterbox experiment implementation.

## Credits

- Development and implementation: Bart Iver van Blokland, [NTNU Visual Computing Lab](https://www.idi.ntnu.no/grupper/vis/)
- Supervision: Theoharis Theoharis, [NTNU Visual Computing Lab](https://www.idi.ntnu.no/grupper/vis/)

If you use (parts of) this library in your research, we kindly ask you reference the papers on which this project is based:

    @article{van2020radial,
      title={Radial intersection count image: A clutter resistant 3D shape descriptor},
      author={van Blokland, Bart Iver and Theoharis, Theoharis},
      journal={Computers \& Graphics},
      volume={91},
      pages={118--128},
      year={2020},
      publisher={Elsevier}
    }

    @article{van2021partial,
      title={Partial 3D object retrieval using local binary QUICCI descriptors and dissimilarity tree indexing},
      author={van Blokland, Bart Iver and Theoharis, Theoharis},
      journal={Computers \& Graphics},
      volume={100},
      pages={32--42},
      year={2021},
      publisher={Elsevier}
    }

## Compiling

The library uses cmake for compilation. You can include it in your project by including it as a subdirectory:

    add_subdirectory(../libShapeDescriptor ${CMAKE_CURRENT_BINARY_DIR}/libShapeDescriptor)
    
It has one configuration option, which defines the resolution of generated spin images, RICI, and QUICCI descriptors:

    add_definitions ( -DspinImageWidthPixels=64 )
    add_subdirectory(../libShapeDescriptor ${CMAKE_CURRENT_BINARY_DIR}/libShapeDescriptor)
    
Also make sure to add "ShapeDescriptor" to the list of linked libraries, and add the 'src' directory to the include path.

This repository contains all necessary libraries to compile the project, except that you need to have the CUDA SDK installed (version 9 or higher).

The library has been tested on Windows and Ubuntu Linux.

## Design

To import all library functionality, only a single header file needs to be included:

```c++
#include <shapeDescriptor/shapeDescriptor.h>
```

The folder structure should hopefully be quite easy to understand. However, it's worth pointing out that any struct will tell you whether it resides in CPU or GPU memory (where applicable):

```c++
// Anything in the 'cpu' namespace lives in RAM and can be accessed directly.
ShapeDescriptor::cpu::array<unsigned int> cpuArray;

// Any struct with the 'gpu' namespace is stored in GPU RAM (VRAM), 
// and must be transferred back and forth explicitly. 
// See the src/utilities/copy directory for functions which can do this for you:
ShapeDescriptor::gpu::array<unsigned int> gpuArray;
```

A number of constants can only be changed at compile time. All such constants can be found in the file 'include/shapeDescriptor/libraryBuildSettings.h'.

## Cookbook

Here are some code samples to help you get up and running quickly.

#### Load Mesh files

Loaders for OBJ, OFF, GLTF, and PLY are included which return a mesh in the format other functions in the library can understand.

```c++
// Load mesh
ShapeDescriptor::cpu::Mesh mesh = ShapeDescriptor::loadMesh("path/to/obj/file.obj",
                                                            ShapeDescriptor::RecomputeNormals::RECOMPUTE_IF_MISSING);

// Free mesh memory
ShapeDescriptor::free(mesh);
```

#### Copy meshes to and from the GPU

All types for which variants exist in CPU and GPU memory can be copied to and from each respective memory using the `ShapeDescriptor::copyToCPU()` and `ShapeDescriptor::copyToGPU()` functions:

```c++
// Load mesh
ShapeDescriptor::cpu::Mesh mesh = ShapeDescriptor::loadMesh("path/to/obj/file.obj",
                                                            ShapeDescriptor::RecomputeNormals::RECOMPUTE_IF_MISSING);

// Copy the mesh to the GPU
ShapeDescriptor::gpu::Mesh gpuMesh = ShapeDescriptor::copyToGPU(mesh);

// And back into CPU memory
ShapeDescriptor::cpu::Mesh returnedMesh = ShapeDescriptor::copyToCPU(gpuMesh);
``` 

Note that each copy operation allocates the required memory automatically. You will therefore need to manually free copies separately. For all meshes, wither they are allocated in CPU or GPU memory you can use:

```c++
// Free all allocated meshes
ShapeDescriptor::free(gpuMesh);
ShapeDescriptor::free(mesh);
ShapeDescriptor::free(returnedMesh);
```

#### Uniformly sample a triangle mesh into a point cloud

Many descriptors work on point clouds instead of triangle meshes, so we've implemented a function which uniformly samples them. Note that since the sampling is performed on the GPU, you first need to copy your mesh into GPU memory.

```c++
size_t sampleCount = 1000000;
size_t randomSeed = 1189998819991197253;
ShapeDescriptor::cpu::Mesh mesh = /* see above */;
ShapeDescriptor::gpu::Mesh gpuMesh = /* see above */;

ShapeDescriptor::cpu::PointCloud sampledPointCloud = ShapeDescriptor::sampleMesh(mesh, sampleCount, randomSeed);
ShapeDescriptor::gpu::PointCloud sampledPointCloud = ShapeDescriptor::sampleMesh(gpuMesh, sampleCount, randomSeed);
```

Note that for a number of functions there exist implementations for the CPU and GPU. The location where the input data resides (that is, CPU or GPU memory) decides where the algorithm is run.

#### Compute descriptors

All descriptors implemented by the library follow the same API. Their parameters take in a scene in the form of a `cpu::Mesh` or `gpu::Mesh`, and a list of vertices for which a descriptor should be computed (referred to as 'origins'). Origins are specified as instances of the `ShapeDescriptor::OrientedPoint`.

Each function returns a `ShapeDescriptor::cpu::array` or `ShapeDescriptor::gpu::array` containing the desired descriptor. The `ShapeDescriptor::copyToCPU()` can be used to transfer any computed descriptors to CPU memory.

For a set of complete example projects, please refer to the [examples directory](https://github.com/bartvbl/libShapeDescriptor/tree/master/examples).


## Roadmap
We intend to make some larger architectural changes for improved ergonomics:
* Remove the need for a separate cpu::array and gpu::array type by replacing cpu::array with std::vector
* Uphold RAII with the Mesh and array types
* Convert the descriptor generator functions to templates rather than relying on compile time constants to determine various dimension parameters
