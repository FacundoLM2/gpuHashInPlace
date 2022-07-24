cmake_minimum_required(VERSION 3.7)
project(cudaProcesses LANGUAGES CXX CUDA)
add_definitions(-std=c++17)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")

find_package(pybind11 REQUIRED)

pybind11_add_module(cudaProcesses pythonEntry.cpp gpuInterface.cu)
