#ifndef __DETECTION_6D_FOUNDATIONPOSE_DECODER_H
#define __DETECTION_6D_FOUNDATIONPOSE_DECODER_H

#include "foundationpose_utils.hpp"

namespace detection_6d {

class FoundationPoseDecoder {
public:
  FoundationPoseDecoder(const int input_poses_num);

  bool DecodeWithMaxScore(int max_score_index,
          const std::vector<Eigen::Matrix4f>& refined_poses,
          Eigen::Matrix4f& out_pose,
          std::shared_ptr<TexturedMeshLoader> mesh_loader);

  bool DecodeInRefine(const Eigen::Matrix4f& refined_pose,
          Eigen::Matrix4f& out_pose,
          std::shared_ptr<TexturedMeshLoader> mesh_loader);

  int GetMaxScoreIndex(void* scores_on_device) noexcept;

  ~FoundationPoseDecoder();

private:
  const int input_poses_num_;
  cudaStream_t cuda_stream_;
};

} // namespace detection_6d


#endif