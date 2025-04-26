#include "detection_6d_foundationpose/foundationpose.hpp"
#include "detection_6d_foundationpose/mesh_loader.hpp"

#include "foundationpose_render.hpp"
#include "foundationpose_sampling.hpp"
#include "foundationpose_utils.hpp"
#include "foundationpose_decoder.cu.hpp"
#include "foundationpose_utils.cu.hpp"

namespace detection_6d {

class FoundationPose : public Base6DofDetectionModel {
public:
  /**
   * @brief 使用多个目标的mesh构建一个FoundationPose实例
   *
   * @param refiner_core refiner的推理核心
   * @param scorer_core scorer的推理核心
   * @param mesh_loaders 注册的三维模型
   * @param intrinsic 相机内参
   * @param input_image_H 输入图像高度，默认480
   * @param input_image_W 输入图像宽度，默认640
   * @param crop_window_H 模型的输入图像高度，默认160
   * @param crop_window_W 模型的输入图像宽度，默认160
   * @param min_depth 有效深度最小值
   * @param max_depth 有效深度最大值
   */
  FoundationPose(std::shared_ptr<inference_core::BaseInferCore>      refiner_core,
                 std::shared_ptr<inference_core::BaseInferCore>      scorer_core,
                 const std::vector<std::shared_ptr<BaseMeshLoader>> &mesh_loaders,
                 const Eigen::Matrix3f                              &intrinsic,
                 const int                                           max_input_image_H = 1080,
                 const int                                           max_input_image_W = 1920,
                 const int                                           crop_window_H     = 160,
                 const int                                           crop_window_W     = 160,
                 const float                                         min_depth         = 0.001);

  bool Register(const cv::Mat     &rgb,
                const cv::Mat     &depth,
                const cv::Mat     &mask,
                const std::string &target_name,
                Eigen::Matrix4f   &out_pose_in_mesh,
                size_t             refine_itr = 1) override;

  bool Track(const cv::Mat         &rgb,
             const cv::Mat         &depth,
             const Eigen::Matrix4f &hyp_pose_in_mesh,
             const std::string     &target_name,
             Eigen::Matrix4f       &out_pose_in_mesh,
             size_t                 refine_itr = 1) override;

private:
  bool CheckInputArguments(const cv::Mat     &rgb,
                           const cv::Mat     &depth,
                           const cv::Mat     &mask,
                           const std::string &target_name);

  using ParsingType = std::unique_ptr<FoundationPosePipelinePackage>;

  bool UploadDataToDevice(const cv::Mat     &rgb,
                          const cv::Mat     &depth,
                          const cv::Mat     &mask,
                          const ParsingType &package);

  bool RefinePreProcess(const ParsingType &package);

  bool RefinePostProcess(const ParsingType &package);

  bool ScorePreprocess(const ParsingType &package);

  bool ScorePostProcess(const ParsingType &package);

  bool TrackPostProcess(const ParsingType &package);

private:
  // 以下参数不对外开放
  // 默认的blob输入名称
  const std::string RENDER_INPUT_BLOB_NAME     = "render_input";
  const std::string TRANSF_INPUT_BLOB_NAME     = "transf_input";
  const std::string REFINE_TRANS_OUT_BLOB_NAME = "trans";
  const std::string REFINE_ROT_OUT_BLOB_NAME   = "rot";
  const float       REFINE_ROT_NORMALIZER      = 0.349065850398865;
  const std::string SCORE_OUTPUT_BLOB_NAME     = "scores";
  // render参数
  const int   score_mode_poses_num_   = 252;
  const int   refine_mode_poses_num_  = 1;
  const float refine_mode_crop_ratio_ = 1.2;
  const float score_mode_crop_ratio_  = 1.1;

private:
  // 以下参数对外开放，通过构造函数传入
  const Eigen::Matrix3f intrinsic_;
  const int             max_input_image_H_;
  const int             max_input_image_W_;
  const int             crop_window_H_;
  const int             crop_window_W_;

  std::shared_ptr<inference_core::BaseInferCore> refiner_core_;
  std::shared_ptr<inference_core::BaseInferCore> scorer_core_;

private:
  // 内部各个模块
  std::unordered_map<std::string, std::shared_ptr<BaseMeshLoader>>         map_name2loaders_;
  std::unordered_map<std::string, std::shared_ptr<FoundationPoseRenderer>> map_name2renderer_;
  std::shared_ptr<FoundationPoseSampler>                                   hyp_poses_sampler_;
};

FoundationPose::FoundationPose(std::shared_ptr<inference_core::BaseInferCore>      refiner_core,
                               std::shared_ptr<inference_core::BaseInferCore>      scorer_core,
                               const std::vector<std::shared_ptr<BaseMeshLoader>> &mesh_loaders,
                               const Eigen::Matrix3f                              &intrinsic,
                               const int   max_input_image_H,
                               const int   max_input_image_W,
                               const int   crop_window_H,
                               const int   crop_window_W,
                               const float min_depth)
    : refiner_core_(refiner_core),
      scorer_core_(scorer_core),
      intrinsic_(intrinsic),
      max_input_image_H_(max_input_image_H),
      max_input_image_W_(max_input_image_W),
      crop_window_H_(crop_window_H),
      crop_window_W_(crop_window_W)
{
  // Check
  auto refiner_blobs_buffer = refiner_core->GetBuffer(true);
  if (refiner_blobs_buffer->GetOuterBlobBuffer(RENDER_INPUT_BLOB_NAME).first == nullptr)
  {
    LOG(ERROR) << "[FoundationPose] Failed to Construct FoundationPose since `renfiner_core` "
               << "do not has a blob named `" << RENDER_INPUT_BLOB_NAME << "`.";
    throw std::runtime_error("[FoundationPose] Failed to Construct FoundationPose");
  }
  if (refiner_blobs_buffer->GetOuterBlobBuffer(TRANSF_INPUT_BLOB_NAME).first == nullptr)
  {
    LOG(ERROR) << "[FoundationPose] Failed to Construct FoundationPose since `renfiner_core` "
               << "do not has a blob named `" << TRANSF_INPUT_BLOB_NAME << "`.";
    throw std::runtime_error("[FoundationPose] Failed to Construct FoundationPose");
  }

  auto scorer_blobs_buffer = scorer_core->GetBuffer(true);
  if (scorer_blobs_buffer->GetOuterBlobBuffer(RENDER_INPUT_BLOB_NAME).first == nullptr)
  {
    LOG(ERROR) << "[FoundationPose] Failed to Construct FoundationPose since `scorer_core` "
               << "do not has a blob named `" << RENDER_INPUT_BLOB_NAME << "`.";
    throw std::runtime_error("[FoundationPose] Failed to Construct FoundationPose");
  }
  if (scorer_blobs_buffer->GetOuterBlobBuffer(TRANSF_INPUT_BLOB_NAME).first == nullptr)
  {
    LOG(ERROR) << "[FoundationPose] Failed to Construct FoundationPose since `scorer_core` "
               << "do not has a blob named `" << TRANSF_INPUT_BLOB_NAME << "`.";
    throw std::runtime_error("[FoundationPose] Failed to Construct FoundationPose");
  }

  // preload modules
  for (const auto &mesh_loader : mesh_loaders)
  {
    const std::string &target_name = mesh_loader->GetName();
    LOG(INFO) << "[FoundationPose] Got target_name : " << target_name;
    map_name2loaders_[target_name] = mesh_loader;
    map_name2renderer_[target_name] =
        std::make_shared<FoundationPoseRenderer>(mesh_loader, intrinsic_, score_mode_poses_num_);
  }

  hyp_poses_sampler_ = std::make_shared<FoundationPoseSampler>(
      max_input_image_H_, max_input_image_W_, min_depth, intrinsic_);
}

bool FoundationPose::CheckInputArguments(const cv::Mat     &rgb,
                                         const cv::Mat     &depth,
                                         const cv::Mat     &mask,
                                         const std::string &target_name)
{
  const int r_rows = rgb.rows, r_cols = rgb.cols;
  const int d_rows = depth.rows, d_cols = depth.cols;
  const int m_rows = mask.empty() ? d_rows : mask.rows, m_cols = mask.empty() ? d_cols : mask.cols;

  if (!(r_rows == d_rows && d_rows == m_rows) || !(r_cols == d_cols && d_cols == m_cols))
  {
    LOG(ERROR) << "[FoundationPose] Got rgb/depth/mask with different size! " << rgb.size << ", "
               << depth.size << ", " << mask.size;
    return false;
  }

  CHECK_STATE(r_rows <= max_input_image_H_ && r_cols <= max_input_image_W_,
              "[FoundationPose] Got rgb/depth/mask with unexpected size !");

  CHECK_STATE(map_name2loaders_.find(target_name) != map_name2loaders_.end(),
              "[FoundationPose] Register Got Invalid `target_name` \
                              which was not provided to FoundationPose instance!!!");

  return true;
}

bool FoundationPose::Register(const cv::Mat     &rgb,
                              const cv::Mat     &depth,
                              const cv::Mat     &mask,
                              const std::string &target_name,
                              Eigen::Matrix4f   &out_pose_in_mesh,
                              size_t             refine_itr)
{
  CHECK_STATE(CheckInputArguments(rgb, depth, mask, target_name),
              "[FoundationPose] `Register` Got invalid arguments!!!");

  auto package           = std::make_unique<FoundationPosePipelinePackage>();
  package->rgb_on_host   = rgb;
  package->depth_on_host = depth;
  package->mask_on_host  = mask;
  package->target_name   = target_name;
  // 将数据传输至device端，并生成xyz_map数据
  MESSURE_DURATION_AND_CHECK_STATE(UploadDataToDevice(rgb, depth, mask, package),
                                   "[FoundationPose] SyncDetect Failed to upload data!!!");

  for (size_t i = 0 ; i < refine_itr ; ++ i) {
    MESSURE_DURATION_AND_CHECK_STATE(
        RefinePreProcess(package),
        "[FoundationPose] SyncDetect Failed to execute RefinePreProcess!!!");

    MESSURE_DURATION_AND_CHECK_STATE(
        refiner_core_->SyncInfer(package->GetInferBuffer()),
        "[FoundationPose] SyncDetect Failed to execute refiner_core_->SyncInfer!!!");

    MESSURE_DURATION_AND_CHECK_STATE(
        RefinePostProcess(package),
        "[FoundationPose] SyncDetect Failed to execute RefinePostProcess!!!");
  }

  MESSURE_DURATION_AND_CHECK_STATE(
      ScorePreprocess(package), "[FoundationPose] SyncDetect Failed to execute ScorePreprocess!!!");

  MESSURE_DURATION_AND_CHECK_STATE(
      scorer_core_->SyncInfer(package->GetInferBuffer()),
      "[FoundationPose] SyncDetect Failed to execute scorer_core_->SyncInfer!!!");

  MESSURE_DURATION_AND_CHECK_STATE(ScorePostProcess(package),
                                   "[FoundationPose] SyncDetect Failed to execute PostProcess!!!");

  out_pose_in_mesh = std::move(package->actual_pose);

  return true;
}

bool FoundationPose::Track(const cv::Mat         &rgb,
                           const cv::Mat         &depth,
                           const Eigen::Matrix4f &hyp_pose_in_mesh,
                           const std::string     &target_name,
                           Eigen::Matrix4f       &out_pose_in_mesh,
                           size_t                 refine_itr)
{
  CHECK_STATE(CheckInputArguments(rgb, depth, cv::Mat(), target_name),
              "[FoundationPose] `Track` Got invalid arguments!!!");

  auto package           = std::make_unique<FoundationPosePipelinePackage>();
  package->rgb_on_host   = rgb;
  package->depth_on_host = depth;
  package->target_name   = target_name;
  package->hyp_poses     = {hyp_pose_in_mesh};
  // 将数据传输至device端，并生成xyz_map数据
  MESSURE_DURATION_AND_CHECK_STATE(UploadDataToDevice(rgb, depth, cv::Mat(), package),
                                   "[FoundationPose] Track Failed to upload data!!!");

  for (size_t i = 0 ; i < refine_itr ; ++ i) {
    MESSURE_DURATION_AND_CHECK_STATE(RefinePreProcess(package),
                                     "[FoundationPose] Track Failed to execute RefinePreProcess!!!");

    MESSURE_DURATION_AND_CHECK_STATE(
        refiner_core_->SyncInfer(package->GetInferBuffer()),
        "[FoundationPose] Track Failed to execute refiner_core_->SyncInfer!!!");

    MESSURE_DURATION_AND_CHECK_STATE(RefinePostProcess(package),
                                     "[Foundation] Track Failed to execute `RefinePostProcess`!!!");
  }

  out_pose_in_mesh = std::move(package->hyp_poses[0]);

  return true;
}

bool FoundationPose::UploadDataToDevice(const cv::Mat     &rgb,
                                        const cv::Mat     &depth,
                                        const cv::Mat     &mask,
                                        const ParsingType &package)
{
  const int input_image_height = rgb.rows, input_image_width = rgb.cols;
  package->input_image_height = input_image_height;
  package->input_image_width  = input_image_width;

  void        *rgb_on_device = nullptr, *depth_on_device = nullptr, *xyz_map_on_device = nullptr;
  const size_t input_image_pixel_num = input_image_height * input_image_width;

  // rgb图像拷贝至device端
  CHECK_CUDA(cudaMalloc(&rgb_on_device, input_image_pixel_num * 3 * sizeof(uint8_t)),
             "[FoundationPose] RefinePreProcess malloc managed `rgb_on_device` failed!!!");
  CHECK_CUDA(cudaMemcpy(rgb_on_device, package->rgb_on_host.data,
                        input_image_pixel_num * 3 * sizeof(uint8_t), cudaMemcpyHostToDevice),
             "[FoundationPose] cudaMemcpy rgb_host -> rgb_device FAILED!!!");

  // depth拷贝至device端
  CHECK_CUDA(cudaMalloc(&depth_on_device, input_image_pixel_num * sizeof(float)),
             "[FoundationPose] RefinePreProcess malloc managed `depth_on_device` failed!!!");
  CHECK_CUDA(cudaMemcpy(depth_on_device, package->depth_on_host.data,
                        input_image_pixel_num * sizeof(float), cudaMemcpyHostToDevice),
             "[FoundationPose] cudaMemcpy depth_host -> depth_device FAILED!!!");

  // 根据depth生成xyz_map，并拷贝至device端
  CHECK_CUDA(cudaMalloc(&xyz_map_on_device, input_image_pixel_num * 3 * sizeof(float)),
             "[FoundationPose] RefinePreProcess malloc managed `xyz_map_on_device` failed!!!");

  convert_depth_to_xyz_map(static_cast<float *>(depth_on_device), input_image_height,
                           input_image_width, static_cast<float *>(xyz_map_on_device),
                           intrinsic_(0, 0), intrinsic_(1, 1), intrinsic_(0, 2), intrinsic_(1, 2),
                           0.001);

  // 输出device端指针，并注册析构过程
  auto func_release_cuda_buffer = [](void *ptr) {
    auto suc = cudaFree(ptr);
    if (suc != cudaSuccess)
    {
      LOG(INFO) << "[FoundationPose] FAILED to free cuda memory!!!";
    }
  };
  package->rgb_on_device     = std::shared_ptr<void>(rgb_on_device, func_release_cuda_buffer);
  package->depth_on_device   = std::shared_ptr<void>(depth_on_device, func_release_cuda_buffer);
  package->xyz_map_on_device = std::shared_ptr<void>(xyz_map_on_device, func_release_cuda_buffer);

  return true;
}

bool FoundationPose::RefinePreProcess(const ParsingType &package)
{
  // 1. sample
  if (package->hyp_poses.empty())
  {
    CHECK_STATE(hyp_poses_sampler_->GetHypPoses(
                    package->depth_on_device.get(), package->mask_on_host.data,
                    package->input_image_height, package->input_image_width, package->hyp_poses),
                "[FoundationPose] Failed to generate hyp poses!!!");
  }

  // 2. render
  if (package->refiner_blobs_buffer == nullptr) {
    package->refiner_blobs_buffer = refiner_core_->GetBuffer(true);
  }
  const auto& refiner_blob_buffer = package->refiner_blobs_buffer;
  // 设置推理前blob的输入位置为device，输出的blob位置为host端
  refiner_blob_buffer->SetBlobBuffer(RENDER_INPUT_BLOB_NAME, DataLocation::DEVICE);
  refiner_blob_buffer->SetBlobBuffer(TRANSF_INPUT_BLOB_NAME, DataLocation::DEVICE);

  auto &refine_renderer     = map_name2renderer_[package->target_name];
  CHECK_STATE(
      refine_renderer->RenderAndTransform(
          package->hyp_poses, package->rgb_on_device.get(), package->depth_on_device.get(),
          package->xyz_map_on_device.get(), package->input_image_height, package->input_image_width,
          refiner_blob_buffer->GetOuterBlobBuffer(RENDER_INPUT_BLOB_NAME).first,
          refiner_blob_buffer->GetOuterBlobBuffer(TRANSF_INPUT_BLOB_NAME).first,
          refine_mode_crop_ratio_),
      "[FoundationPose] Failed to render and transform !!!");
  // 3. 设置推理时形状
  const int input_poses_num = package->hyp_poses.size();
  refiner_blob_buffer->SetBlobShape(RENDER_INPUT_BLOB_NAME,
                                    {input_poses_num, crop_window_H_, crop_window_W_, 6});
  refiner_blob_buffer->SetBlobShape(TRANSF_INPUT_BLOB_NAME,
                                    {input_poses_num, crop_window_H_, crop_window_W_, 6});
  package->infer_buffer         = refiner_blob_buffer;

  return true;
}

bool FoundationPose::RefinePostProcess(const ParsingType &package)
{
  // 获取refiner模型的缓存指针
  const auto &refiner_blob_buffer = package->refiner_blobs_buffer;
  const auto _trans_ptr = refiner_blob_buffer->GetOuterBlobBuffer(REFINE_TRANS_OUT_BLOB_NAME).first;
  const auto _rot_ptr   = refiner_blob_buffer->GetOuterBlobBuffer(REFINE_ROT_OUT_BLOB_NAME).first;
  const float *trans_ptr = static_cast<float *>(_trans_ptr);
  const float *rot_ptr   = static_cast<float *>(_rot_ptr);
  CHECK_STATE(trans_ptr != nullptr, "[FoundationPose] RefinePostProcess got invalid trans_ptr !");
  CHECK_STATE(rot_ptr != nullptr, "[FoundationPose] RefinePostProcess got invalid rot_ptr !");

  // 获取生成的假设位姿
  const auto &hyp_poses = package->hyp_poses;
  const int   poses_num = hyp_poses.size();

  // 获取对应的mesh_loader
  const auto &mesh_loader = map_name2loaders_[package->target_name];

  // transformation 将模型输出的相对位姿转换为绝对位姿
  const float mesh_diameter = mesh_loader->GetMeshDiameter();

  std::vector<Eigen::Vector3f> trans_delta(poses_num);
  std::vector<Eigen::Vector3f> rot_delta(poses_num);
  std::vector<Eigen::Matrix3f> rot_mat_delta(poses_num);

  for (int i = 0; i < poses_num; ++i)
  {
    const size_t offset = i * 3;
    trans_delta[i] << trans_ptr[offset], trans_ptr[offset + 1], trans_ptr[offset + 2];
    trans_delta[i] *= mesh_diameter / 2;

    rot_delta[i] << rot_ptr[offset], rot_ptr[offset + 1], rot_ptr[offset + 2];
    auto normalized_vect = (rot_delta[i].array().tanh() * REFINE_ROT_NORMALIZER).matrix();
    Eigen::AngleAxis rot_delta_angle_axis(normalized_vect.norm(), normalized_vect.normalized());
    rot_mat_delta[i] = rot_delta_angle_axis.toRotationMatrix().transpose();
  }

  std::vector<Eigen::Matrix4f> refine_poses(poses_num);
  for (int i = 0; i < poses_num; ++i)
  {
    refine_poses[i] = hyp_poses[i];
    refine_poses[i].col(3).head(3) += trans_delta[i];

    Eigen::Matrix3f top_left_3x3      = refine_poses[i].block<3, 3>(0, 0);
    Eigen::Matrix3f result_3x3        = rot_mat_delta[i] * top_left_3x3;
    refine_poses[i].block<3, 3>(0, 0) = result_3x3;
  }

  package->hyp_poses = std::move(refine_poses);
  return true;
}

bool FoundationPose::ScorePreprocess(const ParsingType &package)
{
  auto scorer_blob_buffer = scorer_core_->GetBuffer(false);
  // 获取对应的score_renderer
  // 设置推理前后blob输出的位置，这里输入输出都在device端
  scorer_blob_buffer->SetBlobBuffer(RENDER_INPUT_BLOB_NAME, DataLocation::DEVICE);
  scorer_blob_buffer->SetBlobBuffer(TRANSF_INPUT_BLOB_NAME, DataLocation::DEVICE);
  scorer_blob_buffer->SetBlobBuffer(SCORE_OUTPUT_BLOB_NAME, DataLocation::DEVICE);
  auto &score_renderer = map_name2renderer_[package->target_name];
  CHECK_STATE(
      score_renderer->RenderAndTransform(
          package->hyp_poses, package->rgb_on_device.get(), package->depth_on_device.get(),
          package->xyz_map_on_device.get(), package->input_image_height, package->input_image_width,
          scorer_blob_buffer->GetOuterBlobBuffer(RENDER_INPUT_BLOB_NAME).first,
          scorer_blob_buffer->GetOuterBlobBuffer(TRANSF_INPUT_BLOB_NAME).first,
          score_mode_crop_ratio_),
      "[FoundationPose] score_renderer RenderAndTransform Failed!!!");

  package->scorer_blobs_buffer = scorer_blob_buffer;
  package->infer_buffer        = scorer_blob_buffer;

  return true;
}

bool FoundationPose::ScorePostProcess(const ParsingType &package)
{
  const auto &scorer_blob_buffer = package->scorer_blobs_buffer;
  // 获取scorer模型的输出缓存指针
  void *score_ptr = scorer_blob_buffer->GetOuterBlobBuffer(SCORE_OUTPUT_BLOB_NAME).first;

  const auto &refine_poses = package->hyp_poses;
  const int   poses_num    = refine_poses.size();

  // 获取置信度最大的refined_pose
  int max_score_index  = getMaxScoreIndex(nullptr, reinterpret_cast<float *>(score_ptr), poses_num);
  package->actual_pose = refine_poses[max_score_index];

  return true;
}

std::shared_ptr<Base6DofDetectionModel> CreateFoundationPoseModel(
    std::shared_ptr<inference_core::BaseInferCore>      refiner_core,
    std::shared_ptr<inference_core::BaseInferCore>      scorer_core,
    const std::vector<std::shared_ptr<BaseMeshLoader>> &mesh_loaders,
    const Eigen::Matrix3f                              &intrinsic_in_mat,
    const int                                           max_input_image_height,
    const int                                           max_input_image_width)
{
  return std::make_shared<FoundationPose>(refiner_core, scorer_core, mesh_loaders, intrinsic_in_mat,
                                          max_input_image_height, max_input_image_width);
}

} // namespace detection_6d
