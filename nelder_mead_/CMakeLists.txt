cmake_minimum_required(VERSION 3.24)
project(Nelder_Mead)

set(CMAKE_CXX_STANDARD 20)

add_executable(fminsearch_Nelder_Mead nelder_mead.cpp)
include_directories(C:\\PROGRA~0\\include)
if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set (CMAKE_CXX_FLAGS "/kernel /GL /Oi")
endif()
MESSAGE( ${CMAKE_CXX_FLAGS_RELEASE_INIT} )
MESSAGE( STATUS "CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS} )