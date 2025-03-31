#include "foundationpose_decoder.hpp"

#include "foundationpose_decoder.cu.hpp"

namespace detection_6d {

FoundationPoseDecoder::FoundationPoseDecoder(const int input_poses_num)
    : input_poses_num_(input_poses_num)
{
  if (cudaStreamCreate(&cuda_stream_) != cudaSuccess)
  {
    throw std::runtime_error("[FoundationPoseDecoder] Failed to create cuda stream!!!");
  }
}

FoundationPoseDecoder::~FoundationPoseDecoder()
{
  if (cudaStreamDestroy(cuda_stream_) != cudaSuccess)
  {
    LOG(WARNING) << "[FoundationPoseDecoder] Failed to destroy cuda stream!!!";
  }
}

int FoundationPoseDecoder::GetMaxScoreIndex(void *scores_on_device) noexcept
{
  return getMaxScoreIndex(cuda_stream_, reinterpret_cast<float *>(scores_on_device),
                          input_poses_num_);
}

bool FoundationPoseDecoder::DecodeWithMaxScore(int                                 max_score_index,
                                               const std::vector<Eigen::Matrix4f> &refined_poses,
                                               Eigen::Matrix4f                    &out_pose,
                                               std::shared_ptr<TexturedMeshLoader> mesh_loader)
{
  const auto     &best_pose_matrix = refined_poses[max_score_index];
  Eigen::Matrix4f tf_to_center     = Eigen::Matrix4f::Identity();
  tf_to_center.block<3, 1>(0, 3)   = -mesh_loader->GetMeshModelCenter();
  out_pose                         = best_pose_matrix * tf_to_center;
  out_pose                         = out_pose * mesh_loader->GetOrientBounds();
  return true;
}

bool FoundationPoseDecoder::DecodeInRefine(const Eigen::Matrix4f              &refined_pose,
                                           Eigen::Matrix4f                    &out_pose,
                                           std::shared_ptr<TexturedMeshLoader> mesh_loader)
{
  Eigen::Matrix4f tf_to_center   = Eigen::Matrix4f::Identity();
  tf_to_center.block<3, 1>(0, 3) = -mesh_loader->GetMeshModelCenter();
  out_pose                       = refined_pose * tf_to_center;
  out_pose                       = out_pose * mesh_loader->GetOrientBounds();
  return true;
}

} // namespace detection_6d
