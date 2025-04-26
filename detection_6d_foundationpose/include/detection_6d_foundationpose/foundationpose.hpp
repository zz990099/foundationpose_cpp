#pragma once

#include <opencv2/core.hpp>
#include <Eigen/Dense>

#include "detection_6d_foundationpose/mesh_loader.hpp"
#include "deploy_core/base_infer_core.h"

namespace detection_6d {

/**
 * @brief Abstract base class for 6-DoF object detection models
 *
 * @note Implementations should handle both object registration and tracking
 */
class Base6DofDetectionModel {
public:
  /**
   * @brief Register object instance with initial pose estimation
   *
   * @note Implementation must:
   *       - Verify consistent RGB/depth/mask dimensions
   *       - Maintain binding between camera intrinsics and original image size
   *         (resizing requires corresponding intrinsic adjustment)
   *
   * @param rgb Input RGB image (must be in RGB format, convert if using opencv-imread's default
   * BGR)
   * @param depth Depth image (CV_32FC1 format)
   * @param mask Object mask (CV_8UC1 format, positive pixels > 0)
   * @param target_name Object category name (must match construction mapping)
   * @param out_pose_in_mesh Output pose in mesh coordinate frame
   * @param refine_itr Refinement process iteration num
   * @return true Registration successful
   * @return false Registration failed
   */
  virtual bool Register(const cv::Mat     &rgb,
                        const cv::Mat     &depth,
                        const cv::Mat     &mask,
                        const std::string &target_name,
                        Eigen::Matrix4f   &out_pose_in_mesh,
                        size_t             refine_itr = 1) = 0;

  /**
   * @brief Track object pose from subsequent frames (lightweight version of Register)
   *
   * @note Requires prior successful Register call
   *       - Lower accuracy but higher efficiency than Register
   *       - Accepts pose hypotheses from external sources
   *
   * @param rgb Input RGB image (must be in RGB format)
   * @param depth Depth image (CV_32FC1 format)
   * @param hyp_pose_in_mesh Hypothesis pose in mesh frame (from Register or other sources)
   * @param target_name Object category name (must match construction mapping)
   * @param out_pose_in_mesh Output pose in mesh coordinate frame
   * @param refine_itr Refinement process iteration num
   * @return true Tracking successful
   * @return false Tracking failed
   */
  virtual bool Track(const cv::Mat         &rgb,
                     const cv::Mat         &depth,
                     const Eigen::Matrix4f &hyp_pose_in_mesh,
                     const std::string     &target_name,
                     Eigen::Matrix4f       &out_pose_in_mesh,
                     size_t                 refine_itr = 1) = 0;

  /**
   * @brief Virtual destructor for proper resource cleanup
   */
  virtual ~Base6DofDetectionModel() = default;

protected:
  /**
   * @brief Construct a new base 6-DoF detection model
   * @note Protected constructor for abstract base class
   */
  Base6DofDetectionModel() = default;
};

/**
 * @brief Factory function for creating FoundationPose model instances
 *
 * @param refiner_core Refinement inference core handle
 * @param scorer_core Scoring inference core handle
 * @param mesh_loaders Registered 3D mesh loaders
 * @param intrinsic_in_mat Camera intrinsic matrix (3x3, Eigen::Matrix3f)
 * @param max_input_image_height Maximum allowed input image height (pixels)
 * @param max_input_image_width Maximum allowed input image width (pixels)
 * @return std::shared_ptr<Base6DofDetectionModel> Initialized model instance
 *
 * @note Creates an implementation of the 6-DoF detection pipeline using:
 *       - Pose refinement neural network
 *       - Pose scoring network
 *       - Pre-registered 3D mesh models
 *       - Camera calibration parameters
 *       - Input size constraints for memory optimization
 *
 * @warning Input images exceeding max dimensions will cause failure in processing.
 */
std::shared_ptr<Base6DofDetectionModel> CreateFoundationPoseModel(
    std::shared_ptr<inference_core::BaseInferCore>      refiner_core,
    std::shared_ptr<inference_core::BaseInferCore>      scorer_core,
    const std::vector<std::shared_ptr<BaseMeshLoader>> &mesh_loaders,
    const Eigen::Matrix3f                              &intrinsic_in_mat,
    const int                                           max_input_image_height = 1080,
    const int                                           max_input_image_width  = 1920);

} // namespace detection_6d
