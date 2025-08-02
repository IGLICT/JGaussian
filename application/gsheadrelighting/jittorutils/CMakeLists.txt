#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

cmake_minimum_required(VERSION 3.12)

project(DiffRast LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(BUILD_SHARED_LIBS ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

add_library(NetworkOps
	ops/bias_act.cuh
	ops/filtered_lrelu.cuh
	ops/upfirdn2d.cuh
	ops/empty.cu
)

set_target_properties(NetworkOps PROPERTIES CUDA_ARCHITECTURES "70;75;86")

target_include_directories(NetworkOps PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/ops)