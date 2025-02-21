#ifndef __FOUNDATIONPOSE_RENDER_H
#define __FOUNDATIONPOSE_RENDER_H

#include <Eigen/Dense>
#include "cuda.h"
#include "cuda_runtime.h"
#include <cvcuda/OpFlip.hpp>
#include <cvcuda/OpWarpPerspective.hpp>
#include <cvcuda/OpConvertTo.hpp>

#include "nvdiffrast/common/cudaraster/CudaRaster.hpp"
#include "foundationpose_utils.hpp"


namespace detection_6d {


typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMajorMatrix;

class FoundationPoseRenderer {
public:
  FoundationPoseRenderer(std::shared_ptr<TexturedMeshLoader> mesh_loader,
                        const Eigen::Matrix3f& intrinsic,
                        const int input_poses_num,
                        const float crop_ratio = 1.2,
                        const int crop_window_H = 160,
                        const int crop_window_W = 160,
                        const float min_depth = 0.1,
                        const float max_depth = 4.0);

  bool RenderAndTransform(const std::vector<Eigen::Matrix4f>& _poses,
          void* rgb_on_device,
          void* depth_on_device,
          void* xyz_map_on_device,
          int input_image_height,
          int input_image_width,
          void* render_buffer,
          void* transf_buffer);

  ~FoundationPoseRenderer();
private:
  bool RenderProcess(cudaStream_t cuda_stream,
                    const std::vector<Eigen::MatrixXf>& poses,
                    const std::vector<RowMajorMatrix>& tfs,
                    void* poses_on_device,
                    int input_image_height,
                    int input_image_width,
                    void* render_input_dst_ptr);

  bool TransfProcess(cudaStream_t cuda_stream,
                    void* rgb_on_device,
                    void* xyz_map_on_device,
                    int input_image_height,
                    int input_image_width,
                    const std::vector<RowMajorMatrix>& tfs,
                    void* poses_on_device,
                    void* transf_input_dst_ptr);

  bool LoadTexturedMesh();

  bool PrepareBuffer();

  bool TransformVerticesOnCUDA(cudaStream_t stream,
                  const std::vector<Eigen::MatrixXf>& tfs,
                  float* output_buffer) ;

  bool GeneratePoseClipOnCUDA(cudaStream_t stream,
                      float* output_buffer,
                      const std::vector<Eigen::MatrixXf>& poses, 
                      const RowMajorMatrix& bbox2d, 
                      const Eigen::Matrix3f& K, 
                      int rgb_H, int rgb_W);

  bool NvdiffrastRender(cudaStream_t cuda_stream_, 
                        const std::vector<Eigen::MatrixXf>& poses, 
                        const Eigen::Matrix3f& K, 
                        const RowMajorMatrix& bbox2d, 
                        int rgb_H, int rgb_W, int H, int W, 
                        nvcv::Tensor& flip_color_tensor, nvcv::Tensor& flip_xyz_map_tensor);

private:
  //
  const int input_poses_num_;

  // crop window size (model input size)
  const int crop_window_H_;
  const int crop_window_W_;
  const float crop_ratio_; // refine,    score->1.1
  const Eigen::Matrix3f intrinsic_;

  // depth threshold
  const float min_depth_;
  const float max_depth_;


  // mesh
  std::shared_ptr<TexturedMeshLoader> mesh_loader_;
  std::vector<float> vertices_;
  std::vector<float> texcoords_;
  std::vector<int32_t> mesh_faces_;
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> mesh_vertices_;
  int num_vertices_;
  int num_faces_;
  int num_texcoords_;
  float mesh_diameter_;
  int texture_map_height_;
  int texture_map_width_;


  // constants
  const int kNumChannels = 3;
  const size_t kTexcoordsDim = 2;
  const size_t kVertexPoints = 3;
  const size_t kTriangleVertices = 3;
  const size_t kPTMatrixDim = 3;
  // poses位姿变换矩阵维度
  const size_t kTSMatrixDim = 4;

private:
  template<typename T>
  using DeviceBufferUniquePtrType = std::unique_ptr<T, std::function<void(T*)>>;

  DeviceBufferUniquePtrType<float> vertices_device_ {nullptr};
  DeviceBufferUniquePtrType<float> texcoords_device_ {nullptr};
  DeviceBufferUniquePtrType<int32_t> mesh_faces_device_ {nullptr};
  DeviceBufferUniquePtrType<uint8_t> texture_map_device_ {nullptr};
  // nvdiffrast render时相关缓存
  DeviceBufferUniquePtrType<float> pose_clip_device_ {nullptr};
  DeviceBufferUniquePtrType<float> rast_out_device_ {nullptr};
  DeviceBufferUniquePtrType<float> pts_cam_device_ {nullptr};
  DeviceBufferUniquePtrType<float> texcoords_out_device_ {nullptr};
  DeviceBufferUniquePtrType<float> color_device_ {nullptr};
  DeviceBufferUniquePtrType<float> xyz_map_device_ {nullptr};

  // transf 相关缓存
  DeviceBufferUniquePtrType<float> transformed_rgb_device_ {nullptr};
  DeviceBufferUniquePtrType<float> transformed_xyz_map_device_ {nullptr};

  // refine部分输入的poses在过程中是静止的，提供提前计算这部分poses和render结果的功能
  DeviceBufferUniquePtrType<float> input_poses_device_ {nullptr};
  DeviceBufferUniquePtrType<float> render_static_result_buffer_device_ {nullptr};

  std::shared_ptr<CR::CudaRaster> cr_;

  cudaStream_t cuda_stream_render_;
  cudaStream_t cuda_stream_transf_; 

  nvcv::Tensor float_texture_map_tensor_;
};



} // namespace detection_6d




#endif