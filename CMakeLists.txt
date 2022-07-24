cmake_minimum_required(VERSION 3.7)
project(cudaProcesses LANGUAGES CXX CUDA)
add_definitions(-std=c++17)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")

find_package(pybind11 REQUIRED)

pybind11_add_module(cudaProcesses pythonEntry.cpp gpuInterface.cu)

# add_executable(main main.cpp tiff.cpp gpuInterface.cpp )

# add_library(tiff tiff.cpp)
# add_library(gpuInterface gpuInterface.cu)
# include_directories("${PROJECT_SOURCE_DIR}/")

# add_definitions( -DMAGICKCORE_QUANTUM_DEPTH=8 )
# add_definitions( -DMAGICKCORE_HDRI_ENABLE=1 )
# find_package(ImageMagick COMPONENTS Magick++)

# include_directories(${ImageMagick_INCLUDE_DIRS})
# target_link_libraries(testImageMagick ${ImageMagick_LIBRARIES})
# target_link_libraries(cudaProcesses PRIVATE pybind11::module ${ImageMagick_LIBRARIES})