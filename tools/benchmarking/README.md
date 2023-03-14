# How to use benchmarking tool

## Building the project

1. After having cloned the project, do `git submodule update --init` in the root folder.
2. To be able to build the project you need to have `cmake` and `Ninja` installed.
3. If these are installed you first create the _build_ folder if it doesn't already exist. `mkdir build`.
4. Following this, navigate to the _build_ folder and execute the `cmake .. -DCMAKE_BUILD_TYPE=Release -G Ninja` command and then do `Ninja`.
5. This will create a _benchmarking_ executable in the _build_ folder which you can run with `./benchmarking`

## Running the project

You are required to add two different paths to the objects you want to compare, which can only be in the OBJ file format. To accompany the objects, you should also add their metadata, but this isn't required. Additionally, you have some optional inputs. You can specify which descriptor algorithm and which distance algorithm it should use to compare the similarity, by inputting their corresponding number which is given in the help screen.

```
benchmarking: Compare how similar two objects are (only OBJ file support)
usage: benchmarking
    -o, --original-object=<string>
		Original object.
    -c, --comparison-object=<string>
		Object to compare to the original.
    -f, --objects-folder=<string>
		Folder consisting of sub-directories with all the different objects and their metadata
    -n, --originals-folder=<string>
		Folder name with all the original objects (for example, RecalculatedNormals)
    -F, --compare-folder=<string>
		If you only want to compare the originals to a specific folder (for example, ObjectsWithHoles)
    -m, --metadata=<string>
		Path to metadata describing which vertecies that are changed
    -p, --output-path=<string>
		Path to the output
    -a, --descriptor-algorithm=<int>
		Which descriptor algorithm to use [0 for radial-intersection-count-images, 1 for quick-intersection-count-change-images ...will add more:)]
    -d, --distance-algorithm=<int>
		Which distance algorithm to use [0 for euclidian, ...will add more:)]
    -t, --hardware-type=<string>
		cpu or gpu (gpu is default, as cpu doesn't support all the descriptors)
    -h, --help
		Show help
```

Example:

```
./benchmarking -o=/Users/jonathanbrooks/masteroppgaven/objects/RecalculatedNormals/0000.obj -c=/Users/jonathanbrooks/masteroppgaven/objects/ResizedObjects/0000/0000.obj -m=/Users/jonathanbrooks/masteroppgaven/objects/ResizedObjects/0000/0000.txt

0.134604
```

The output is a number between 0 and 1 declaring how equal the objects are.

## Folder structure

As this code isn't very dynamic ( :) ) it expect some sort of folder structure when you want to compare two folders to each other. Each folder should first consist of a set of categories, atleast one, and then inside each category you put your object files inside a folder with the same name. The most important part is that an object's folder has the same name in both directories.

```
.
└── OriginalObjects/
    └── Category/
        └── Object1/
            └── Object1.obj

.
└── ComparisonObjects/
    └── Category/
        └── Object1/
            ├── Object1.obj
            └── Object1.txt
```

To compare two folders to each other you write:

```
./benchmarking -f=/Folder/where/objects/folders/are/... -n=OriginalObjects -F=ComparisonObjects -p=/Where/to/put/your/output/...
```

## Metadata definition

The metadata is just a long list of numbers. Where each numbers placement in the list (starting from 0) is the corresponding vertex index for the comparison object. You can also use ~ to indicate that the vertex has been deleted somehow, and that it should be skipped.

```
0
1
2
~
3
...
```
