cmake_minimum_required(VERSION 3.24)
project(cuda_db LANGUAGES CXX)

# CUDA language support is enabled once kernel work begins (a later
# milestone); left out for now so this builds on machines without nvcc.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(cpp)
