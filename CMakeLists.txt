cmake_minimum_required(VERSION 3.15)
project(autodiff)

set(CMAKE_CXX_STANDARD 17)

add_executable(autodiff main.cpp autodiff/Variable.cpp autodiff/Variable.h autodiff/Tensor.cpp autodiff/Tensor.h autodiff/Tape.cpp autodiff/Tape.h autodiff/Dense.cpp autodiff/Dense.h)