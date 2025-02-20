#ifndef __FOUNDATIONPOSE_SAMPLING_H
#define __FOUNDATIONPOSE_SAMPLING_H

#include <vector>
#include <Eigen/Dense>

#include "foundationpose_utils.hpp"

namespace detection_6d {

class FoundationPoseSampler {
public:
  FoundationPoseSampler(const int max_input_image_H,
                        const int max_input_image_W,
                        const float min_depth,
                        const float max_depth,
                        const Eigen::Matrix3f& intrinsic);

  bool GetHypPoses(void* _depth_on_device,
                  void* _mask_on_host,
                  int input_image_height,
                  int input_image_width,
                  std::vector<Eigen::Matrix4f>& out_hyp_poses);

  ~FoundationPoseSampler();

private:
  const int max_input_image_H_;
  const int max_input_image_W_;
  const float min_depth_;
  const Eigen::Matrix3f intrinsic_;
  cudaStream_t cuda_stream_;

private:
  float* erode_depth_buffer_device_;
  float* bilateral_depth_buffer_device_;
  std::vector<float> bilateral_depth_buffer_host_;
  const std::vector<Eigen::Matrix4f> pre_compute_rotations_;
};

} // namespace detection_6d

#endif