#pragma once

#include <cstdint>
#include <limits>

#include "cuda.h"
#include "cuda_runtime.h"


void convert_depth_to_xyz_map(cudaStream_t cuda_stream, const float* depth_on_device, int input_image_height,
    int input_image_width, float* xyz_map_on_device, const float fx, const float fy, const float dx, const float dy, const float min_depth);

void convert_depth_to_xyz_map(const float* depth_on_device, int input_image_height,
    int input_image_width, float* xyz_map_on_device, const float fx, const float fy, const float dx, const float dy, const float min_depth);