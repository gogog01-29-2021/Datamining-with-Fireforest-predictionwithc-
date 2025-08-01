# C++ Data Mining Utilities

## Requirements

- C++17 or later
- [Eigen](https://eigen.tuxfamily.org/) for matrix operations and statistics
- (Optional) [matplotlib-cpp](https://github.com/lava/matplotlib-cpp) for plotting

## Build (all .cpp files)

1. Install Eigen (header-only):
   - Download from https://eigen.tuxfamily.org/
   - Extract and place the Eigen folder somewhere accessible
2. Compile all C++ files in the main and subfolders:
   ```sh
   g++ -std=c++17 -I /path/to/eigen *.cpp 250314Shenfeng/*.cpp 250327KYK\ dataminingweek4/*.cpp -o main
   ```
   Replace `/path/to/eigen` with the path to the Eigen `include` directory.

## Run

```sh
./main
```

## Using make

You can use a simple Makefile. Place this in your project root:

```
CXX = g++
CXXFLAGS = -std=c++17 -I /path/to/eigen
SOURCES = $(wildcard *.cpp) $(wildcard 250314Shenfeng/*.cpp) $(wildcard 250327KYK\ dataminingweek4/*.cpp)
OBJECTS = $(SOURCES:.cpp=.o)
TARGET = main

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJECTS)

clean:
	rm -f $(OBJECTS) $(TARGET)
```

Replace `/path/to/eigen` with your Eigen path. Then run:

```sh
make
./main
```

## Notes
- This code demonstrates basic statistics, quantiles, IQR, outlier detection, correlation, one-hot encoding, and scaling in C++.
- For plotting, see [matplotlib-cpp](https://github.com/lava/matplotlib-cpp) or export data to Python/Excel for visualization.
- For deep learning, see [LibTorch](https://pytorch.org/cppdocs/).
- Each Python file is translated to a C++ file with the same name and `.cpp` extension, in the same folder structure.
- All logic is in the corresponding `.cpp` files for each original Python file.
=======
# Datamining-with-Fireforest-predictionwithc-
>>>>>>> a0110fcff2fd8215e06712258ba58833b909ae90
