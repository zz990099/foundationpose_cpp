#include "foundationpose_sampling.hpp"
#include "foundationpose_sampling.cu.hpp"

#include <map>
#include <glog/logging.h>
#include <glog/log_severity.h>

#include <opencv2/core.hpp>

namespace detection_6d {

typedef Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMajorMatrix8u;

// A helper function to create a vertex from a point
int AddVertex(const Eigen::Vector3f &p, std::vector<Eigen::Vector3f> &vertices)
{
  vertices.push_back(p.normalized());
  return vertices.size() - 1;
}

// A helper function to create a face from three indices
void AddFace(int i, int j, int k, std::vector<Eigen::Vector3i> &faces)
{
  faces.emplace_back(i, j, k);
}

// A helper function to get the middle point of two vertices
int GetMiddlePoint(int                           i,
                   int                           j,
                   std::vector<Eigen::Vector3f> &vertices,
                   std::map<int64_t, int>       &cache)
{
  // check if the edge (i, j) has been processed before
  bool    first_is_smaller = i < j;
  int64_t smaller          = first_is_smaller ? i : j;
  int64_t greater          = first_is_smaller ? j : i;
  int64_t key              = (smaller << 32) + greater;

  auto it = cache.find(key);
  if (it != cache.end())
  {
    return it->second;
  }

  // if not cached, create a new vertex
  Eigen::Vector3f p1    = vertices[i];
  Eigen::Vector3f p2    = vertices[j];
  Eigen::Vector3f pm    = (p1 + p2) / 2.0;
  int             index = AddVertex(pm, vertices);
  cache[key]            = index;
  return index;
}

// A function to generate an icosphere
// Initial triangle values could found from https://sinestesia.co/blog/tutorials/python-icospheres/
std::vector<Eigen::Vector3f> GenerateIcosphere(unsigned int n_views)
{
  std::map<int64_t, int>       cache;
  std::vector<Eigen::Vector3f> vertices;
  std::vector<Eigen::Vector3i> faces;

  // create 12 vertices
  float t = (1.0 + std::sqrt(5.0)) / 2.0; // the golden ratio
  AddVertex(Eigen::Vector3f(-1, t, 0), vertices);
  AddVertex(Eigen::Vector3f(1, t, 0), vertices);
  AddVertex(Eigen::Vector3f(-1, -t, 0), vertices);
  AddVertex(Eigen::Vector3f(1, -t, 0), vertices);
  AddVertex(Eigen::Vector3f(0, -1, t), vertices);
  AddVertex(Eigen::Vector3f(0, 1, t), vertices);
  AddVertex(Eigen::Vector3f(0, -1, -t), vertices);
  AddVertex(Eigen::Vector3f(0, 1, -t), vertices);
  AddVertex(Eigen::Vector3f(t, 0, -1), vertices);
  AddVertex(Eigen::Vector3f(t, 0, 1), vertices);
  AddVertex(Eigen::Vector3f(-t, 0, -1), vertices);
  AddVertex(Eigen::Vector3f(-t, 0, 1), vertices);

  // create 20 faces
  AddFace(0, 11, 5, faces);
  AddFace(0, 5, 1, faces);
  AddFace(0, 1, 7, faces);
  AddFace(0, 7, 10, faces);
  AddFace(0, 10, 11, faces);
  AddFace(1, 5, 9, faces);
  AddFace(5, 11, 4, faces);
  AddFace(11, 10, 2, faces);
  AddFace(10, 7, 6, faces);
  AddFace(7, 1, 8, faces);
  AddFace(3, 9, 4, faces);
  AddFace(3, 4, 2, faces);
  AddFace(3, 2, 6, faces);
  AddFace(3, 6, 8, faces);
  AddFace(3, 8, 9, faces);
  AddFace(4, 9, 5, faces);
  AddFace(2, 4, 11, faces);
  AddFace(6, 2, 10, faces);
  AddFace(8, 6, 7, faces);
  AddFace(9, 8, 1, faces);

  // subdivide each face into four smaller faces
  while (vertices.size() < n_views)
  {
    std::vector<Eigen::Vector3i> new_faces;
    for (const auto &face : faces)
    {
      int a = face[0];
      int b = face[1];
      int c = face[2];

      int ab = GetMiddlePoint(a, b, vertices, cache);
      int bc = GetMiddlePoint(b, c, vertices, cache);
      int ca = GetMiddlePoint(c, a, vertices, cache);

      AddFace(a, ab, ca, new_faces);
      AddFace(b, bc, ab, new_faces);
      AddFace(c, ca, bc, new_faces);
      AddFace(ab, bc, ca, new_faces);
    }
    faces = new_faces;
  }
  return std::move(vertices);
}

float RotationGeodesticDistance(const Eigen::Matrix3f &R1, const Eigen::Matrix3f &R2)
{
  float cos = ((R1 * R2.transpose()).trace() - 1) / 2.0;
  cos       = std::max(std::min(cos, 1.0f), -1.0f);
  return std::acos(cos);
}

std::vector<Eigen::Matrix4f> ClusterPoses(float                         angle_diff,
                                          float                         dist_diff,
                                          std::vector<Eigen::Matrix4f> &poses_in,
                                          std::vector<Eigen::Matrix4f> &symmetry_tfs)
{
  std::vector<Eigen::Matrix4f> poses_out;
  poses_out.push_back(poses_in[0]);
  const float radian_thres = angle_diff / 180.0 * M_PI;

  for (unsigned int i = 1; i < poses_in.size(); i++)
  {
    bool            is_new   = true;
    Eigen::Matrix4f cur_pose = poses_in[i];

    for (const auto &cluster : poses_out)
    {
      Eigen::Vector3f t0 = cluster.block(0, 3, 3, 1);
      Eigen::Vector3f t1 = cur_pose.block(0, 3, 3, 1);
      if ((t0 - t1).norm() >= dist_diff)
      {
        continue;
      }
      // Remove symmetry
      for (const auto &tf : symmetry_tfs)
      {
        Eigen::Matrix4f cur_pose_tmp = cur_pose * tf;
        float           rot_diff =
            RotationGeodesticDistance(cur_pose_tmp.block(0, 0, 3, 3), cluster.block(0, 0, 3, 3));
        if (rot_diff < radian_thres)
        {
          is_new = false;
          break;
        }
      }
      if (!is_new)
      {
        break;
      }
    }

    if (is_new)
    {
      poses_out.push_back(poses_in[i]);
    }
  }
  return std::move(poses_out);
}

std::vector<Eigen::Matrix4f> SampleViewsIcosphere(unsigned int n_views)
{
  auto vertices = GenerateIcosphere(n_views);
  std::vector<Eigen::Matrix4f, std::allocator<Eigen::Matrix4f>> cam_in_obs(
      vertices.size(), Eigen::Matrix4f::Identity(4, 4));
  for (unsigned int i = 0; i < vertices.size(); i++)
  {
    cam_in_obs[i].block<3, 1>(0, 3) = vertices[i];
    Eigen::Vector3f up(0, 0, 1);
    Eigen::Vector3f z_axis = -cam_in_obs[i].block<3, 1>(0, 3);
    z_axis.normalize();

    Eigen::Vector3f x_axis = up.cross(z_axis);
    if (x_axis.isZero())
    {
      x_axis << 1, 0, 0;
    }
    x_axis.normalize();
    Eigen::Vector3f y_axis = z_axis.cross(x_axis);
    y_axis.normalize();
    cam_in_obs[i].block<3, 1>(0, 0) = x_axis;
    cam_in_obs[i].block<3, 1>(0, 1) = y_axis;
    cam_in_obs[i].block<3, 1>(0, 2) = z_axis;
  }
  return std::move(cam_in_obs);
}

/**
 * @brief 创建一个`n_views`面体，返回它的顶点位姿集合
 *
 * @param n_views 默认40
 * @param inplane_step 默认60
 * @return std::vector<Eigen::Matrix4f>
 */
std::vector<Eigen::Matrix4f> MakeRotationGrid(unsigned int n_views = 40, int inplane_step = 60)
{
  auto cam_in_obs = SampleViewsIcosphere(n_views);

  std::vector<Eigen::Matrix4f> rot_grid;
  for (unsigned int i = 0; i < cam_in_obs.size(); i++)
  {
    for (double inplane_rot = 0; inplane_rot < 360; inplane_rot += inplane_step)
    {
      Eigen::Matrix4f cam_in_ob = cam_in_obs[i];
      auto            R_inplane = Eigen::Affine3f::Identity();
      R_inplane.rotate(Eigen::AngleAxisf(0, Eigen::Vector3f::UnitX()))
          .rotate(Eigen::AngleAxisf(0, Eigen::Vector3f::UnitY()))
          .rotate(Eigen::AngleAxisf(inplane_rot * M_PI / 180.0f, Eigen::Vector3f::UnitZ()));

      cam_in_ob                 = cam_in_ob * R_inplane.matrix();
      Eigen::Matrix4f ob_in_cam = cam_in_ob.inverse();
      rot_grid.push_back(ob_in_cam);
    }
  }

  std::vector<Eigen::Matrix4f> symmetry_tfs = std::vector<Eigen::Matrix4f>();
  symmetry_tfs.push_back(Eigen::Matrix4f::Identity());
  ClusterPoses(30.0, 99999.0, rot_grid, symmetry_tfs);
  return std::move(rot_grid);
}

/**
 * @brief 根据深度图、掩码、相机内参来估计目标物体的近似三维中心
 *
 * @param depth 深度图
 * @param mask 掩码
 * @param K 相机内参
 * @param min_depth 有效最小深度
 * @param center 输出三维中心
 * @return true
 * @return false
 */
bool GuessTranslation(const Eigen::MatrixXf  &depth,
                      const RowMajorMatrix8u &mask,
                      const Eigen::Matrix3f  &K,
                      float                   min_depth,
                      Eigen::Vector3f        &center)
{
  // Find the indices where mask is positive
  std::vector<int> vs, us;
  for (int i = 0; i < mask.rows(); i++)
  {
    for (int j = 0; j < mask.cols(); j++)
    {
      if (mask(i, j) > 0)
      {
        vs.push_back(i);
        us.push_back(j);
      }
    }
  }
  CHECK_STATE(!us.empty(), "[FoundationposeSampling] Mask is all zero.");

  float uc =
      (*std::min_element(us.begin(), us.end()) + *std::max_element(us.begin(), us.end())) / 2.0;
  float vc =
      (*std::min_element(vs.begin(), vs.end()) + *std::max_element(vs.begin(), vs.end())) / 2.0;

  Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> valid =
      (mask.array() > 0) && (depth.array() >= min_depth);
  CHECK_STATE(valid.any(), "[FoundationposeSampling] No valid value in mask.");

  std::vector<float> valid_depth;
  for (int i = 0; i < valid.rows(); i++)
  {
    for (int j = 0; j < valid.cols(); j++)
    {
      if (valid(i, j))
      {
        valid_depth.push_back(depth(i, j));
      }
    }
  }
  std::sort(valid_depth.begin(), valid_depth.end());
  int   n = valid_depth.size();
  float zc =
      (n % 2 == 0) ? (valid_depth[n / 2 - 1] + valid_depth[n / 2]) / 2.0 : valid_depth[n / 2];

  center = K.inverse() * Eigen::Vector3f(uc, vc, 1) * zc;
  return true;
}

FoundationPoseSampler::FoundationPoseSampler(const int              max_input_image_H,
                                             const int              max_input_image_W,
                                             const float            min_depth,
                                             const Eigen::Matrix3f &intrinsic)
    : max_input_image_H_(max_input_image_H),
      max_input_image_W_(max_input_image_W),
      min_depth_(min_depth),
      intrinsic_(intrinsic),
      pre_compute_rotations_(MakeRotationGrid())
{
  CHECK_CUDA_THROW(cudaStreamCreate(&cuda_stream_),
                   "[FoundationPoseSampler] Failed to create cuda stream!!");

  CHECK_CUDA_THROW(cudaMalloc(&erode_depth_buffer_device_,
                              max_input_image_H_ * max_input_image_W_ * sizeof(float)),
                   "[FoundationPoseSampler] Failed to malloc cuda memory of `erode_depth`");

  CHECK_CUDA_THROW(cudaMalloc(&bilateral_depth_buffer_device_,
                              max_input_image_H_ * max_input_image_W_ * sizeof(float)),
                   "[FoundationPoseSampler] Failed to malloc cuda memory of `bilateral_depth`");

  bilateral_depth_buffer_host_.resize(max_input_image_H_ * max_input_image_W_);

  LOG(INFO) << "[FoundationPoseSampler] Pre-computed rotations, size : "
            << pre_compute_rotations_.size();
}

FoundationPoseSampler::~FoundationPoseSampler()
{
  if (cudaStreamDestroy(cuda_stream_))
  {
    LOG(WARNING) << "[FoundationPoseSampler] Failed to destroy cuda stream !";
  }

  if (cudaFree(erode_depth_buffer_device_) != cudaSuccess)
  {
    LOG(WARNING) << "[FoundationPoseSampler] Failed to free `erode_depth` buffer on device !";
  }
  if (cudaFree(bilateral_depth_buffer_device_) != cudaSuccess)
  {
    LOG(WARNING) << "[FoundationPoseSampler] Failed to free `bilateral_depth` buffer on device !";
  }
}

bool FoundationPoseSampler::GetHypPoses(void                         *_depth_on_device,
                                        void                         *_mask_on_host,
                                        int                           input_image_height,
                                        int                           input_image_width,
                                        std::vector<Eigen::Matrix4f> &out_hyp_poses)
{
  if (_depth_on_device == nullptr || _mask_on_host == nullptr)
  {
    throw std::invalid_argument("[FoudationPoseSampler] Got INVALID depth/mask ptr on device!!!");
  }
  // 1. 生成基于多面体的初始假设位姿
  out_hyp_poses = pre_compute_rotations_;
  // 2. 优化depth深度图
  float *depth_on_device = static_cast<float *>(_depth_on_device);
  // 2.1 depth腐蚀操作
  erode_depth(cuda_stream_, depth_on_device, erode_depth_buffer_device_, input_image_height,
              input_image_width);
  // 2.2 depth双边滤波操作
  bilateral_filter_depth(cuda_stream_, erode_depth_buffer_device_, bilateral_depth_buffer_device_,
                         input_image_height, input_image_width);
  // 2.3 拷贝到host端缓存
  cudaMemcpyAsync(bilateral_depth_buffer_host_.data(), bilateral_depth_buffer_device_,
                  input_image_height * input_image_width * sizeof(float), cudaMemcpyDeviceToHost,
                  cuda_stream_);

  // 2.4 同步cuda流
  CHECK_CUDA(cudaStreamSynchronize(cuda_stream_),
             "[FoundationPoseSampling] cudaStreamSync `cuda_stream_` FAILED!!!");

  // 3. 基于depth、mask估计目标物三维中心
  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      bilateral_filter_depth_host(bilateral_depth_buffer_host_.data(), input_image_height,
                                  input_image_width);
  Eigen::Map<Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mask_host(
      static_cast<uint8_t *>(_mask_on_host), input_image_height, input_image_width);

  Eigen::Vector3f center;
  CHECK_STATE(
      GuessTranslation(bilateral_filter_depth_host, mask_host, intrinsic_, min_depth_, center),
      "[FoundationPose Sampling] Failed to GuessTranslation!!!");

  LOG(INFO) << "[FoundationPose Sampling] Center: " << center;

  // 4. 把三维中心放到变换矩阵内
  for (auto &pose : out_hyp_poses)
  {
    pose.block<3, 1>(0, 3) = center;
  }

  return true;
}

} // namespace detection_6d
