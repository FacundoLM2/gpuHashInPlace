# gpuHashInPlace
Sketch for a cuda implementation of reassignment of values in an array according to a pre-computed hash table. The python script runs a multi-thread CPU function to accomplish this and compared with a CUDA GPU implementation. GPU implementation of hash table taken from: https://github.com/nosferalatu/SimpleGPUHashTable

# compilation instructions:

## install nvcc compiler 

download and install Nvidia toolkit and compilers from: https://developer.nvidia.com/nvidia-hpc-sdk-downloads
select the compiler version compatible with your Nvidia driver installation. Make sure nvcc and nvprof are callable from terminal. You may need to add the following line to .bashrc:

    export LD_LIBRARY_PATH="/usr/local/cuda-<version>/targets/x86_64-linux/include:$LD_LIBRARY_PATH"

## pybind11 is required to create python bindings

install pybind11 for python binding (sudo install at system level python)

    sudo pip install "pybind11[global]"

## install required python packages:

    pip install -r requirements.txt

## compile with cmake:

    mkdir build
    cd build
    cmake ..
    make

output library is compiled to build/cudaProcesses.cpython-38-x86_64-linux-gnu.so (naming convention will depend on your hardware configuration) 

## return to root directory and run python script using Nvidia profiler

    cd ..
    nvprof python3 main.py


