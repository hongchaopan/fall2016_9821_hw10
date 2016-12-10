cmake_minimum_required(VERSION 3.6)
project(MTH9821_HW10)

set(CMAKE_CXX_STANDARD 11)

include_directories("/usr/local/Eigen3")


set(SOURCE_FILES
        main.cpp Heat_Discrete_PDE_DIV.hpp Heat_Discrete_PDE_DIV.cpp
        LinearSolver.hpp
        LinearSolver.cpp
        )
add_executable(MTH9821_HW10 ${SOURCE_FILES})