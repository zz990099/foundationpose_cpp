#ifndef __FOUNDATIONPOSE_H
#define __FOUNDATIONPOSE_H

#include <opencv2/core.hpp>
#include <Eigen/Dense>

#include "deploy_core/base_infer_core.h"

namespace detection_6d {

class Base6DofDetectionModel {
public:
  /**
   * @brief 实现的接口应支持动态输入尺寸，并检查rgb/depth/mask尺寸是否一致.
   *
   * @note
   * 相机内参与原始图像的尺寸是绑定的，不可在外部直接对图像进行resize操作，若需要进行resize，应对内参intrinsic同样处理
   *
   * @param rgb rgb图像，必须是`rgb`格式，从opencv-imread读取的图像默认是bgr格式，需经过转换
   * @param depth 获取的深度图像，cv::Mat数据格式为CV_32F1
   * @param mask 目标物的mask图像，cv::Mat数据格式为CV_8UC1，positive的像素值大于0即可
   * @param target_name 目标物的名称，与构建时提供的name->mesh_path映射一致
   * @param out_pose 输出位姿
   * @return true
   * @return false
   */
  virtual bool Register(const cv::Mat     &rgb,
                        const cv::Mat     &depth,
                        const cv::Mat     &mask,
                        const std::string &target_name,
                        Eigen::Matrix4f   &out_pose) = 0;
  /**
   * @brief
   * 从第二帧开始的跟踪过程，是`Register`的轻量化版本，精度稍低但推理效率非常高，调用前必须先调用`Register`
   *
   * @param rgb rgb图像，必须是`rgb`格式，从opencv-imread读取的图像默认是bgr格式，需经过转换
   * @param depth 获取的深度图像，cv::Mat数据格式为CV_32F1
   * @param target_name 目标物的名称，与构建时提供的name->mesh_path映射一致
   * @param out_pose 输出位姿
   * @return true
   * @return false
   */
  virtual bool Track(const cv::Mat     &rgb,
                     const cv::Mat     &depth,
                     const std::string &target_name,
                     Eigen::Matrix4f   &out_pose) = 0;

  /**
   * @brief 获取某个输入mesh目标的三维尺寸(辅助功能，用户也可自己计算)
   *
   * @return Eigen::Vector3f
   */
  virtual Eigen::Vector3f GetObjectDimension(const std::string &target_name) const
  {
    throw std::runtime_error("[Base6DofDetectionModel] GetOjbectDimension NOT Implemented yet!!!");
  };

  virtual ~Base6DofDetectionModel() = default;

protected:
  Base6DofDetectionModel() = default;
};

/**
 * @brief 创建一个`FoundationPose`实例
 *
 * @param refiner_core refiner推理核心
 * @param scorer_core scorer推理核心
 * @param mesh_file_path 使用的三维模型`mesh_file`路径
 * @param texture_file_path 使用的三维模型外观特征图像路径
 * @param intrinsic_in_vec 相机内参矩阵，std::vector<float>形式，`row_major`格式
 * @return std::shared_ptr<Base6DofDetectionModel>
 */
std::shared_ptr<Base6DofDetectionModel> CreateFoundationPoseModel(
    std::shared_ptr<inference_core::BaseInferCore> refiner_core,
    std::shared_ptr<inference_core::BaseInferCore> scorer_core,
    const std::string                             &target_name,
    const std::string                             &mesh_file_path,
    const std::string                             &texture_file_path,
    const std::vector<float>                      &intrinsic_in_vec);

/**
 * @brief 创建一个`FoundationPose`实例
 *
 * @param refiner_core refiner推理核心
 * @param scorer_core scorer推理核心
 * @param mesh_file_path 使用的三维模型`mesh_file`路径
 * @param texture_file_path 使用的三维模型外观特征图像路径
 * @param intrinsic_in_mat 相机内参矩阵，Eigen::Matrix3f格式
 * @return std::shared_ptr<Base6DofDetectionModel>
 */
std::shared_ptr<Base6DofDetectionModel> CreateFoundationPoseModel(
    std::shared_ptr<inference_core::BaseInferCore> refiner_core,
    std::shared_ptr<inference_core::BaseInferCore> scorer_core,
    const std::string                             &target_name,
    const std::string                             &mesh_file_path,
    const std::string                             &texture_file_path,
    const Eigen::Matrix3f                         &intrinsic_in_mat);

/**
 * @brief 创建一个`FoundationPose`实例
 *
 * @param refiner_core refiner推理核心
 * @param scorer_core scorer推理核心
 * @param meshes 多个目标的mesh/texture路径map: [name] -> [mesh_file_path, texture_file_path]，
 *             键值[name]在后续检测过程中用于辨别特定种类目标，**保持一致**
 * @param intrinsic_in_vec 相机内参矩阵，std::vector<float>形式，`row_major`格式
 * @return std::shared_ptr<Base6DofDetectionModel>
 */
std::shared_ptr<Base6DofDetectionModel> CreateFoundationPoseModel(
    std::shared_ptr<inference_core::BaseInferCore>                              refiner_core,
    std::shared_ptr<inference_core::BaseInferCore>                              scorer_core,
    const std::unordered_map<std::string, std::pair<std::string, std::string>> &meshes,
    const std::vector<float>                                                   &intrinsic_in_vec);

/**
 * @brief 创建一个`FoundationPose`实例
 *
 * @param refiner_core refiner推理核心
 * @param scorer_core scorer推理核心
 * @param meshes 多个目标的mesh/texture路径map: [name] -> [mesh_file_path, texture_file_path]，
 *               键值[name]在后续检测过程中用于辨别特定种类目标，**保持一致**
 * @param intrinsic_in_mat 相机内参矩阵，Eigen::Matrix3f格式
 * @return std::shared_ptr<Base6DofDetectionModel>
 */
std::shared_ptr<Base6DofDetectionModel> CreateFoundationPoseModel(
    std::shared_ptr<inference_core::BaseInferCore>                              refiner_core,
    std::shared_ptr<inference_core::BaseInferCore>                              scorer_core,
    const std::unordered_map<std::string, std::pair<std::string, std::string>> &meshes,
    const Eigen::Matrix3f                                                      &intrinsic_in_mat);

} // namespace detection_6d

#endif
