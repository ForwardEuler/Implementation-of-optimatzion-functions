cmake_minimum_required(VERSION 3.22)
project(c_plus_plus)

#set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 20)

include_directories(C:\\PROGRA~0\\include)
add_executable(c_plus_plus pso_parallel_cpu.cpp Jprand.h Jprand.cpp)
#set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp /O2")
#set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -openmp")
set (CMAKE_CXX_FLAGS "/EHsc /openmp /O2 /GL")
MESSAGE( STATUS "CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS} )
