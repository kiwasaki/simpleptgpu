cmake_minimum_required( VERSION 3.17 )
project( simpleptgpu CUDA )

set( CMAKE_CUDA_STANDARD 14 )
include_directories( /usr/local/cuda-11.0/samples/common/inc/ )
add_executable( simpleptgpu main.cu ray.hpp scene.hpp sphere.hpp intersection.hpp )

set_target_properties(
        simpleptgpu
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)