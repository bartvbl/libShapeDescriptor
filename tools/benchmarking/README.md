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
		[required] Original object.
    -c, --comparison-object=<string>
		[required] Object to compare to the original.
    -m, --metadata=<string>
		Path to metadata describing which vertecies that are changed
    -a, --descriptor-algorithm=<int>
		Which descriptor algorithm to use [0 for radial-intersection-count-images, ...will add more:)]
    -d, --distance-algorithm=<int>
		Which distance algorithm to use [0 for euclidian, ...will add more:)]
    -h, --help
		Show help
```

Example:

```
./benchmarking -o=/Users/jonathanbrooks/masteroppgaven/objects/RecalculatedNormals/0000.obj -c=/Users/jonathanbrooks/masteroppgaven/objects/ResizedObjects/0000/0000.obj -m=/Users/jonathanbrooks/masteroppgaven/objects/ResizedObjects/0000/0000.txt

0.134604
```

The output is a number between 0 and 1 declaring how equal the objects are.

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
