# libShapeDescriptor

This library contains efficient GPU implementations for computing and comparing the following descriptors efficiently on the GPU:

- Spin Image (Johnson et al., 1999)
- 3D Shape Context (Frome et al., 2004)
- Fast Point Feature Histogram (Rusu et al., 2009)
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
      volume="91",
      pages="118--128",
      year={2020},
      publisher={Elsevier}
    }
    
    @article{van2020indexing,
      title={An Indexing Scheme and Descriptor for 3D Object Retrieval Based on Local Shape Querying},
      author={van Blokland, Bart Iver and Theoharis, Theoharis},
      journal={Computers \& Graphics},
      volume="92",
      pages="55-66",
      year={2020},
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

The folder structure should hopefully be quite easy to understand. However, it's worth pointing out that any struct will tell you whether it resides in CPU or GPU memory:

```c++
// Anything in the 'cpu' namespace lives in RAM and can be accessed directly.
ShapeDescriptor::cpu::array<unsigned int> cpuArray;

// Any struct with the 'gpu' namespace is stored in GPU RAM (VRAM), 
// and must be transferred back and forth explicitly. 
// See the src/utilities/copy directory for functions which can do this for you:
ShapeDescriptor::gpu::array<unsigned int> gpuArray;
```

A number of constants can only be changed at compile time. All such constants can be found in the file 'src/shapeDescriptor/libraryBuildSettings.h'.

## Cookbook

Here are some code samples to help you get up and running quickly.

#### Load Mesh files

Loaders for OBJ, OFF, and PLY are included which return a mesh in the format other functions in the library can understand.

```c++
// Load mesh
const bool recomputeNormals = false;
ShapeDescriptor::cpu::Mesh mesh = ShapeDescriptor::utilities::loadMesh("path/to/obj/file.obj", recomputeNormals);

// Free mesh memory
ShapeDescriptor::free::mesh(mesh);
```

#### Copy meshes to and from the GPU

The src/utilities/copy directory contains a number of functions which can copy all relevant data structures to and from the GPU.

```c++
// Load mesh
const bool recomputeNormals = false;
ShapeDescriptor::cpu::Mesh mesh = ShapeDescriptor::utilities::loadMesh("path/to/obj/file.obj", recomputeNormals);

// Copy the mesh to the GPU
ShapeDescriptor::gpu::Mesh gpuMesh = ShapeDescriptor::copy::hostMeshToDevice(mesh);

// And back into CPU memory
ShapeDescriptor::cpu::Mesh returnedMesh = ShapeDescriptor::copy::deviceMeshToHost(gpuMesh);
``` 

Note that each copy operation allocates the required memory automatically. You will therefore need to manually free copies separately. For all meshes, wither they are allocated in CPU or GPU memory you can use:

```c++
// Free all allocated meshes
ShapeDescriptor::free::mesh(gpuMesh);
ShapeDescriptor::free::mesh(mesh);
ShapeDescriptor::free::mesh(returnedMesh);
```

#### Uniformly sample a triangle mesh into a point cloud

Many descriptors work on point clouds instead of triangle meshes, so we've implemented a function which uniformly samples them. Note that since the sampling is performed on the GPU, you first need to copy your mesh into GPU memory.

```c++
size_t sampleCount = 1000000;
size_t randomSeed = 1189998819991197253;
ShapeDescriptor::gpu::Mesh gpuMesh = /* see above */;
ShapeDescriptor::gpu::PointCloud sampledPointCloud = ShapeDescriptor::utilities::sampleMesh(gpuMesh, sampleCount, randomSeed);
```

#### Compute descriptors

All descriptors implemented by the library follow the same API. Their parameters take in a scene, and a list of vertices for which a descriptor should be computed (referred to as 'origins'). Origins are specified as instances of the ShapeDescriptor::gpu::OrientedPoint.

Each function returns a ShapeDescriptor::gpu::array containing the desired descriptor. There's a function in the src/shapeDescriptor/utilities/copy directory for transferring them to CPU memory.

Here's a complete example for computing a single RICI descriptor:

```c++
// Load mesh
ShapeDescriptor::cpu::Mesh mesh = ShapeDescriptor::utilities::loadMesh("path/to/obj/file.obj", false);
    
// Store it on the GPU
ShapeDescriptor::gpu::Mesh gpuMesh = ShapeDescriptor::copy::hostMeshToDevice(mesh);

// Define and upload descriptor origins
ShapeDescriptor::gpu::array<ShapeDescriptor::gpu::OrientedPoint> descriptorOrigins;
descriptorOrigins.length = 1;
descriptorOrigins.content = new OrientedPoint[1];
descriptorOrigins.content[0].vertex = {0.5, 0.5, 0.5};
descriptorOrigins.content[0].normal = {0, 0, 1};

ShapeDescriptor::gpu::array<ShapeDescriptor::gpu::OrientedPoint> gpuDescriptorOrigins = 
    ShapeDescriptor::copy::hostArrayToDevice(descriptorOrigins);

// Compute the descriptor(s)
float supportRadius = 1.0;
ShapeDescriptor::gpu::array<ShapeDescriptor::RICIDescriptor> descriptors = 
    ShapeDescriptor::gpu::generateRadialIntersectionCountImages(
            gpuMesh,
            gpuDescriptorOrigins,
            supportRadius);
            
// Copy descriptors to RAM
ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor> hostDescriptors =
            ShapeDescriptor::copy::deviceArrayToHost(descriptors);
                
// Do something with descriptors here

// Free memory
ShapeDescriptor::free::array(descriptorOrigins);
ShapeDescriptor::free::array(hostDescriptors);
ShapeDescriptor::free::array(gpuDescriptorOrigins);
ShapeDescriptor::free::array(descriptors);
ShapeDescriptor::free::mesh(mesh);
ShapeDescriptor::free::mesh(gpuMesh);
```
