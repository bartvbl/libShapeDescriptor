# How to use benchmarking tool

## Building the project

1. After having cloned the project, do `git submodule update --init` in the root folder.
2. To be able to build the project you need to have `cmake` and `Ninja` installed.
3. If these are installed you first create the _build_ folder if it doesn't already exist. `mkdir build`.
4. Following this, navigate to the _build_ folder and execute the `cmake .. -DCMAKE_BUILD_TYPE=Release -G Ninja` command and then do `Ninja`.
5. This will create a _benchmarking_ executable in the _build_ folder which you can run with `./benchmarking`

## Running the project

You are required to add two different paths to the objects you want to compare, which can only be in the OBJ file format. Additionally, you have some optional inputs. You can specify which descriptor algorithm and which distance algorithm it should use to compare the similarity, by inputting their corresponding number which is given in the help screen.

```
    -o, --original-object=<string>
		[required] Original object.
    -c, --comparison-object=<string>
		[required] Object to compare to the original.
    -a, --descriptor-algorithm=<int>
		Which descriptor algorithm to use [0 for radial-intersection-count-images, ...will add more:)]
    -d, --distance-algorithm=<int>
		Which distance algorithm to use [0 for euclidian, ...will add more:)]
    -h, --help
		Show help
```

Example:

```
./benchmarking -o='/Users/jonathanbrooks/masteroppgaven/objects/shark.obj' -c='/Users/jonathanbrooks/masteroppgaven/objects/shark.obj'

1
```

The output is a number between 0 and 1 declaring how equal the objects are.
