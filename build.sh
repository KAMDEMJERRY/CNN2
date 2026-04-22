echo "Building CNN..."

# Check if build directory exists, if not create it
if [ ! -d "build" ]; then
    mkdir build
fi

# Navigate to build directory
cd build

# Run CMake to configure the project
cmake ..

# Compile the project
make

# Navigate back to src directory
cd src

# Run the CNN executable
./CNN