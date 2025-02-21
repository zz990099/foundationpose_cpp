#include "foundationpose_utils.cu.hpp"


__global__ void convert_depth_to_xyz_map_kernel(const float* depth_on_device, int input_image_height,
    int input_image_width, float* xyz_map_on_device, const float fx, const float fy, const float dx, const float dy, const float min_depth)
{
  const int row_idx = threadIdx.y + blockIdx.y * blockDim.y;
  const int col_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (row_idx >= input_image_height || col_idx >= input_image_width) return;

  const int pixel_idx = row_idx * input_image_width + col_idx;

  const float depth = depth_on_device[pixel_idx];
  if (depth < min_depth) return;

  const float x = (col_idx - dx) * depth / fx;
  const float y = (row_idx - dy) * depth / fy;
  const float z = depth;

  float* this_pixel_xyz = xyz_map_on_device + pixel_idx * 3;
  this_pixel_xyz[0] = x;
  this_pixel_xyz[1] = y;
  this_pixel_xyz[2] = z;
}

static uint16_t ceil_div(uint16_t numerator, uint16_t denominator) {
  uint32_t accumulator = numerator + denominator - 1;
  return accumulator / denominator + 1;
}


void convert_depth_to_xyz_map(cudaStream_t cuda_stream, const float* depth_on_device, int input_image_height,
    int input_image_width, float* xyz_map_on_device, const float fx, const float fy, const float dx, const float dy, const float min_depth)
{
  dim3 blockSize = {32, 32};
  dim3 gridSize = {ceil_div(input_image_width, 32), ceil_div(input_image_height, 32)};

  convert_depth_to_xyz_map_kernel<<<gridSize, blockSize, 0, cuda_stream>>>(
      depth_on_device, input_image_height, input_image_width, xyz_map_on_device, fx, fy, dx, dy, min_depth);
}


void convert_depth_to_xyz_map(const float* depth_on_device, int input_image_height,
    int input_image_width, float* xyz_map_on_device, const float fx, const float fy, const float dx, const float dy, const float min_depth)
{
  dim3 blockSize = {32, 32};
  dim3 gridSize = {ceil_div(input_image_width, 32), ceil_div(input_image_height, 32)};

  convert_depth_to_xyz_map_kernel<<<gridSize, blockSize, 0>>>(
      depth_on_device, input_image_height, input_image_width, xyz_map_on_device, fx, fy, dx, dy, min_depth);
}
