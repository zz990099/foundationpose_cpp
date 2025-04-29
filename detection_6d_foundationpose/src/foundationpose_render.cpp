#include "foundationpose_render.hpp"

#include <fstream>
#include "foundationpose_render.cu.hpp"
#include "foundationpose_utils.hpp"

namespace detection_6d {

void saveFloatsToFile(const float *data, size_t N, const std::string &filename)
{
  std::ofstream outFile(filename, std::ios::binary);
  if (!outFile)
  {
    std::cerr << "Error opening file for writing." << std::endl;
    return;
  }
  outFile.write(reinterpret_cast<const char *>(data), N * sizeof(float));
  outFile.close();
}

// From OpenCV camera (cvcam) coordinate system to the OpenGL camera (glcam) coordinate system
const Eigen::Matrix4f kGLCamInCVCam =
    (Eigen::Matrix4f(4, 4) << 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1).finished();

RowMajorMatrix ComputeTF(float left, float right, float top, float bottom, Eigen::Vector2i out_size)
{
  left   = std::round(left);
  right  = std::round(right);
  top    = std::round(top);
  bottom = std::round(bottom);

  RowMajorMatrix tf = Eigen::MatrixXf::Identity(3, 3);
  tf(0, 2)          = -left;
  tf(1, 2)          = -top;

  RowMajorMatrix new_tf = Eigen::MatrixXf::Identity(3, 3);
  new_tf(0, 0)          = out_size(0) / (right - left);
  new_tf(1, 1)          = out_size(1) / (bottom - top);

  auto result = new_tf * tf;
  return result;
}

std::vector<RowMajorMatrix> ComputeCropWindowTF(const std::vector<Eigen::MatrixXf> &poses,
                                                const Eigen::MatrixXf              &K,
                                                Eigen::Vector2i                     out_size,
                                                float                               crop_ratio,
                                                float                               mesh_diameter)
{
  // Compute the tf batch from the left, right, top, and bottom coordinates
  int             B = poses.size();
  float           r = mesh_diameter * crop_ratio / 2;
  Eigen::MatrixXf offsets(5, 3);
  offsets << 0, 0, 0, r, 0, 0, -r, 0, 0, 0, r, 0, 0, -r, 0;

  std::vector<RowMajorMatrix> tfs;
  for (int i = 0; i < B; i++)
  {
    auto            block     = poses[i].block<3, 1>(0, 3).transpose();
    Eigen::MatrixXf pts       = block.replicate(offsets.rows(), 1).array() + offsets.array();
    Eigen::MatrixXf projected = (K * pts.transpose()).transpose();
    Eigen::MatrixXf uvs =
        projected.leftCols(2).array() / projected.rightCols(1).replicate(1, 2).array();
    Eigen::MatrixXf center = uvs.row(0);

    float radius = std::abs((uvs - center.replicate(uvs.rows(), 1)).rightCols(1).maxCoeff());
    float left   = center(0, 0) - radius;
    float right  = center(0, 0) + radius;
    float top    = center(0, 1) - radius;
    float bottom = center(0, 1) + radius;

    tfs.push_back(ComputeTF(left, right, top, bottom, out_size));
  }
  return tfs;
}

/**
 * @brief
 *
 * @param output
 * @param pts
 * @param tfs
 * @return true
 * @return false
 */
bool TransformPts(std::vector<RowMajorMatrix>        &output,
                  const Eigen::MatrixXf              &pts,
                  const std::vector<Eigen::MatrixXf> &tfs)
{
  // Get the dimensions of the inputs
  int rows     = pts.rows();
  int cols     = pts.cols();
  int tfs_size = tfs.size();
  CHECK_STATE(tfs_size != 0, "[FoundationposeRender] The transfomation matrix is empty! ");

  CHECK_STATE(tfs[0].cols() == tfs[0].rows(),
              "[FoundationposeRender] The transfomation matrix has different rows and cols! ");

  int dim = tfs[0].rows();
  CHECK_STATE(cols == dim - 1,
              "[FoundationposeRender] The dimension of pts and tf are not match! ");

  for (int i = 0; i < tfs_size; i++)
  {
    RowMajorMatrix transformed_matrix;
    transformed_matrix.resize(rows, dim - 1);
    auto submatrix = tfs[i].block(0, 0, dim - 1, dim - 1);
    auto last_col  = tfs[i].block(0, dim - 1, dim - 1, 1);

    // Apply the transformation to the points
    for (int j = 0; j < rows; j++)
    {
      auto new_row              = submatrix * pts.row(j).transpose() + last_col;
      transformed_matrix.row(j) = new_row.transpose();
    }
    output.push_back(transformed_matrix);
  }

  // Return the result vector
  return true;
}

bool ConstructBBox2D(RowMajorMatrix &bbox2d, const std::vector<RowMajorMatrix> &tfs, int H, int W)
{
  Eigen::MatrixXf bbox2d_crop(2, 2);
  bbox2d_crop << 0.0, 0.0, W - 1, H - 1;

  std::vector<Eigen::MatrixXf> inversed_tfs;
  // Inverse tfs before transform
  for (size_t i = 0; i < tfs.size(); i++)
  {
    inversed_tfs.push_back(tfs[i].inverse());
  }

  std::vector<RowMajorMatrix> bbox2d_ori_vec;
  auto                        suc = TransformPts(bbox2d_ori_vec, bbox2d_crop, inversed_tfs);
  if (!suc)
  {
    LOG(ERROR) << "[FoundationposeRender] Failed to transform the 2D bounding box";
    return suc;
  }

  for (size_t i = 0; i < bbox2d_ori_vec.size(); i++)
  {
    bbox2d.row(i) =
        Eigen::Map<Eigen::RowVectorXf>(bbox2d_ori_vec[i].data(), bbox2d_ori_vec[i].size());
  }
  return true;
}

bool ProjectMatrixFromIntrinsics(Eigen::Matrix4f       &proj_output,
                                 const Eigen::Matrix3f &K,
                                 int                    height,
                                 int                    width,
                                 float                  znear         = 0.1,
                                 float                  zfar          = 100.0,
                                 std::string            window_coords = "y_down")
{
  int   x0 = 0;
  int   y0 = 0;
  int   w  = width;
  int   h  = height;
  float nc = znear;
  float fc = zfar;

  float depth = fc - nc;
  float q     = -(fc + nc) / depth;
  float qn    = -2 * (fc * nc) / depth;

  // Get the projection matrix from camera K matrix
  if (window_coords == "y_up")
  {
    proj_output << 2 * K(0, 0) / w, -2 * K(0, 1) / w, (-2 * K(0, 2) + w + 2 * x0) / w, 0, 0,
        -2 * K(1, 1) / h, (-2 * K(1, 2) + h + 2 * y0) / h, 0, 0, 0, q, qn, 0, 0, -1, 0;
  } else if (window_coords == "y_down")
  {
    proj_output << 2 * K(0, 0) / w, -2 * K(0, 1) / w, (-2 * K(0, 2) + w + 2 * x0) / w, 0, 0,
        2 * K(1, 1) / h, (2 * K(1, 2) - h + 2 * y0) / h, 0, 0, 0, q, qn, 0, 0, -1, 0;
  } else
  {
    LOG(ERROR) << "[FoundationposeRender] The window coordinates should be y_up or y_down";
    return false;
  }

  return true;
}

void WrapImgPtrToNHWCTensor(
    uint8_t *input_ptr, nvcv::Tensor &output_tensor, int N, int H, int W, int C)
{
  nvcv::TensorDataStridedCuda::Buffer output_buffer;
  output_buffer.strides[3] = sizeof(uint8_t);
  output_buffer.strides[2] = C * output_buffer.strides[3];
  output_buffer.strides[1] = W * output_buffer.strides[2];
  output_buffer.strides[0] = H * output_buffer.strides[1];
  output_buffer.basePtr    = reinterpret_cast<NVCVByte *>(input_ptr);

  nvcv::TensorShape::ShapeType shape{N, H, W, C};
  nvcv::TensorShape            tensor_shape{shape, "NHWC"};
  nvcv::TensorDataStridedCuda  output_data(tensor_shape, nvcv::TYPE_U8, output_buffer);
  output_tensor = nvcv::TensorWrapData(output_data);
}

void WrapFloatPtrToNHWCTensor(
    float *input_ptr, nvcv::Tensor &output_tensor, int N, int H, int W, int C)
{
  nvcv::TensorDataStridedCuda::Buffer output_buffer;
  output_buffer.strides[3] = sizeof(float);
  output_buffer.strides[2] = C * output_buffer.strides[3];
  output_buffer.strides[1] = W * output_buffer.strides[2];
  output_buffer.strides[0] = H * output_buffer.strides[1];
  output_buffer.basePtr    = reinterpret_cast<NVCVByte *>(input_ptr);

  nvcv::TensorShape::ShapeType shape{N, H, W, C};
  nvcv::TensorShape            tensor_shape{shape, "NHWC"};
  nvcv::TensorDataStridedCuda  output_data(tensor_shape, nvcv::TYPE_F32, output_buffer);
  output_tensor = nvcv::TensorWrapData(output_data);
}

FoundationPoseRenderer::FoundationPoseRenderer(std::shared_ptr<BaseMeshLoader> mesh_loader,
                                               const Eigen::Matrix3f          &intrinsic,
                                               const int                       input_poses_num,
                                               const int                       crop_window_H,
                                               const int                       crop_window_W,
                                               const float                     min_depth,
                                               const float                     max_depth)
    : mesh_loader_(mesh_loader),
      intrinsic_(intrinsic),
      input_poses_num_(input_poses_num),
      crop_window_H_(crop_window_H),
      crop_window_W_(crop_window_W),
      min_depth_(min_depth),
      max_depth_(max_depth)
{
  if (cudaStreamCreate(&cuda_stream_render_) != cudaSuccess ||
      cudaStreamCreate(&cuda_stream_transf_) != cudaSuccess)
  {
    throw std::runtime_error("[FoundationPose Renderer] Failed to create cuda stream!!!");
  }

  // 1. load mesh file
  bool load_mesh_suc = LoadTexturedMesh();
  if (!load_mesh_suc)
  {
    throw std::runtime_error("[FoundationPose Renderer] Failed to load textured mesh!!!");
  }

  // 2. prepare device buffer
  bool prepare_buf_suc = PrepareBuffer();
  if (!prepare_buf_suc)
  {
    throw std::runtime_error("[FoundationPose Renderer] Failed to prepare buffer!!!");
  }
}

FoundationPoseRenderer::~FoundationPoseRenderer()
{
  if (cudaStreamDestroy(cuda_stream_render_) != cudaSuccess ||
      cudaStreamDestroy(cuda_stream_transf_) != cudaSuccess)
  {
    LOG(WARNING) << "[FoundationPoseRenderer] Failed to destroy cuda stream !";
  }
}

bool FoundationPoseRenderer::PrepareBuffer()
{
  // nvdiffrast render 用到的缓存以及渲染器
  size_t pose_clip_size = num_vertices_ * (kVertexPoints + 1) * input_poses_num_ * sizeof(float);
  size_t pts_cam_size   = num_vertices_ * kVertexPoints * input_poses_num_ * sizeof(float);
  size_t diffuse_intensity_size = num_vertices_ * input_poses_num_ * sizeof(float);
  size_t diffuse_intensity_map_size =
      input_poses_num_ * crop_window_H_ * crop_window_W_ * sizeof(float);
  size_t rast_out_size =
      input_poses_num_ * crop_window_H_ * crop_window_W_ * (kVertexPoints + 1) * sizeof(float);
  size_t color_size =
      input_poses_num_ * crop_window_H_ * crop_window_W_ * kNumChannels * sizeof(float);
  size_t xyz_map_size =
      input_poses_num_ * crop_window_H_ * crop_window_W_ * kNumChannels * sizeof(float);
  size_t texcoords_out_size =
      input_poses_num_ * crop_window_H_ * crop_window_W_ * kTexcoordsDim * sizeof(float);

  // nvdiffrast render时相关缓存
  float *_pose_clip_device;
  float *_rast_out_device;
  float *_pts_cam_device;
  float *_diffuse_intensity_device;
  float *_diffuse_intensity_map_device;
  float *_texcoords_out_device;
  float *_color_device;
  float *_xyz_map_device;
  float *_render_crop_rgb_tensor_device;
  float *_render_crop_xyz_map_tensor_device;

  // transf 相关缓存
  float   *_transformed_rgb_device;
  float   *_transformed_xyz_map_device;
  uint8_t *_transformed_crop_rgb_tensor_device;

  // 输入的假设位姿
  float *_input_poses_device;

  // render用到的缓存
  CHECK_CUDA(cudaMalloc(&_pose_clip_device, pose_clip_size),
             "[FoundationPoseRenderer] cudaMalloc `_pose_clip_device` FAILED!!!");
  pose_clip_device_ =
      DeviceBufferUniquePtrType<float>(_pose_clip_device, CudaMemoryDeleter<float>());

  CHECK_CUDA(cudaMalloc(&_rast_out_device, rast_out_size),
             "[FoundationPoseRenderer] cudaMalloc `_rast_out_device` FAILED!!!");
  rast_out_device_ = DeviceBufferUniquePtrType<float>(_rast_out_device, CudaMemoryDeleter<float>());

  CHECK_CUDA(cudaMalloc(&_pts_cam_device, pts_cam_size),
             "[FoundationPoseRenderer] cudaMalloc `_pts_cam_device` FAILED!!!");
  pts_cam_device_ = DeviceBufferUniquePtrType<float>(_pts_cam_device, CudaMemoryDeleter<float>());

  CHECK_CUDA(cudaMalloc(&_diffuse_intensity_device, diffuse_intensity_size),
             "[FoundationPoseRenderer] cudaMalloc `_diffuse_intensity_device` FAILED!!!");
  diffuse_intensity_device_ =
      DeviceBufferUniquePtrType<float>(_diffuse_intensity_device, CudaMemoryDeleter<float>());

  CHECK_CUDA(cudaMalloc(&_diffuse_intensity_map_device, diffuse_intensity_map_size),
             "[FoundationPoseRenderer] cudaMalloc `_diffuse_intensity_map_device` FAILED!!!");
  diffuse_intensity_map_device_ =
      DeviceBufferUniquePtrType<float>(_diffuse_intensity_map_device, CudaMemoryDeleter<float>());

  CHECK_CUDA(cudaMalloc(&_texcoords_out_device, texcoords_out_size),
             "[FoundationPoseRenderer] cudaMalloc `_texcoords_out_device` FAILED!!!");
  texcoords_out_device_ =
      DeviceBufferUniquePtrType<float>(_texcoords_out_device, CudaMemoryDeleter<float>());

  CHECK_CUDA(cudaMalloc(&_color_device, color_size),
             "[FoundationPoseRenderer] cudaMalloc `_color_device` FAILED!!!");
  color_device_ = DeviceBufferUniquePtrType<float>(_color_device, CudaMemoryDeleter<float>());

  CHECK_CUDA(cudaMalloc(&_xyz_map_device, xyz_map_size),
             "[FoundationPoseRenderer] cudaMalloc `_xyz_map_device` FAILED!!!");
  xyz_map_device_ = DeviceBufferUniquePtrType<float>(_xyz_map_device, CudaMemoryDeleter<float>());

  CHECK_CUDA(cudaMalloc(&_render_crop_rgb_tensor_device, color_size),
             "[FoundationPoseRenderer] cudaMalloc `_color_device` FAILED!!!");
  render_crop_rgb_tensor_device_ =
      DeviceBufferUniquePtrType<float>(_render_crop_rgb_tensor_device, CudaMemoryDeleter<float>());

  CHECK_CUDA(cudaMalloc(&_render_crop_xyz_map_tensor_device, xyz_map_size),
             "[FoundationPoseRenderer] cudaMalloc `_xyz_map_device` FAILED!!!");
  render_crop_xyz_map_tensor_device_ = DeviceBufferUniquePtrType<float>(
      _render_crop_xyz_map_tensor_device, CudaMemoryDeleter<float>());

  // transf 用到的缓存
  const size_t device_buffer_byte_size =
      input_poses_num_ * crop_window_H_ * crop_window_W_ * kNumChannels * sizeof(float);

  CHECK_CUDA(cudaMalloc(&_transformed_xyz_map_device, device_buffer_byte_size),
             "[FoundationPoseRenderer] cudaMalloc `_transformed_xyz_map_device` FAILED!!!");
  transformed_xyz_map_device_ =
      DeviceBufferUniquePtrType<float>(_transformed_xyz_map_device, CudaMemoryDeleter<float>());

  CHECK_CUDA(cudaMalloc(&_transformed_rgb_device, device_buffer_byte_size),
             "[FoundationPoseRenderer] cudaMalloc `_transformed_rgb_device` FAILED!!!");
  transformed_rgb_device_ =
      DeviceBufferUniquePtrType<float>(_transformed_rgb_device, CudaMemoryDeleter<float>());

  const size_t crop_rgb_byte_size = 1 * crop_window_H_ * crop_window_W_ * kNumChannels;
  CHECK_CUDA(cudaMalloc(&_transformed_crop_rgb_tensor_device, crop_rgb_byte_size),
             "[FoundationPoseRenderer] cudaMalloc `_transformed_crop_rgb_tensor_device` FAILED!!!");
  transformed_crop_rgb_tensor_device_ = DeviceBufferUniquePtrType<uint8_t>(
      _transformed_crop_rgb_tensor_device, CudaMemoryDeleter<uint8_t>());

  // poses 的device缓存
  CHECK_CUDA(cudaMalloc(&_input_poses_device,
                        input_poses_num_ * kTSMatrixDim * kTSMatrixDim * sizeof(float)),
             "[FoundationPoseRenderer] cudaMalloc `_input_poses_device` FAILED!!!");
  input_poses_device_ =
      DeviceBufferUniquePtrType<float>(_input_poses_device, CudaMemoryDeleter<float>());

  cr_ = std::make_unique<CR::CudaRaster>();

  return true;
}

bool FoundationPoseRenderer::LoadTexturedMesh()
{
  const auto &mesh_model_center   = mesh_loader_->GetMeshModelCenter();
  const auto &mesh_vertices       = mesh_loader_->GetMeshVertices();
  const auto &mesh_vertex_normals = mesh_loader_->GetMeshVertexNormals();
  const auto &mesh_texcoords      = mesh_loader_->GetMeshTextureCoords();
  const auto &mesh_faces          = mesh_loader_->GetMeshTriangleFaces();
  const auto &rgb_texture_map     = mesh_loader_->GetTextureMap();
  mesh_diameter_                  = mesh_loader_->GetMeshDiameter();

  std::vector<float> vertex_normals;

  // Walk through each of the mesh's vertices
  for (unsigned int v = 0; v < mesh_vertices.size(); v++)
  {
    vertices_.push_back(mesh_vertices[v][0] - mesh_model_center[0]);
    vertices_.push_back(mesh_vertices[v][1] - mesh_model_center[1]);
    vertices_.push_back(mesh_vertices[v][2] - mesh_model_center[2]);

    vertex_normals.push_back(mesh_vertex_normals[v][0]);
    vertex_normals.push_back(mesh_vertex_normals[v][1]);
    vertex_normals.push_back(mesh_vertex_normals[v][2]);

    // Check if the mesh has texture coordinates
    texcoords_.push_back(mesh_texcoords[v][0]);
    texcoords_.push_back(1 - mesh_texcoords[v][1]);
  }

  // Walk through each of the mesh's faces (a face is a mesh its triangle)
  for (unsigned int f = 0; f < mesh_faces.size(); f++)
  {
    for (unsigned int i = 0; i < kTriangleVertices; i++)
    {
      mesh_faces_.push_back(mesh_faces[f][i]);
    }
  }

  // Load the texture map
  CHECK_STATE(rgb_texture_map.isContinuous(),
              "[FoundationposeRender] Texture map is not continuous");

  CHECK_STATE(rgb_texture_map.channels() == kNumChannels,
              "[FoundationposeRender] Recieved texture map has" +
                  std::to_string(rgb_texture_map.channels()) + " number of channels, expected " +
                  std::to_string(kNumChannels));

  texture_map_height_ = rgb_texture_map.rows;
  texture_map_width_  = rgb_texture_map.cols;

  // The number of vertices is the size of the vertices array divided by 3 (since it's x,y,z)
  num_vertices_ = vertices_.size() / kVertexPoints;
  // The number of texture coordinates is the size of the texcoords array divided by 2 (since it's
  // u,v)
  num_texcoords_ = texcoords_.size() / kTexcoordsDim;
  // The number of faces is the size of the faces array divided by 3 (since each face has 3 edges)
  num_faces_ = mesh_faces_.size() / kTriangleVertices;

  mesh_vertices_ =
      Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
          vertices_.data(), num_vertices_, 3);

  // Allocate device memory for mesh data
  size_t vertices_size  = vertices_.size() * sizeof(float);
  size_t faces_size     = mesh_faces_.size() * sizeof(int32_t);
  size_t texcoords_size = texcoords_.size() * sizeof(float);

  float   *_vertices_device;
  float   *_vertex_normals_device;
  float   *_texcoords_device;
  int32_t *_mesh_faces_device;
  uint8_t *_texture_map_device;

  CHECK_CUDA(cudaMalloc(&_vertices_device, vertices_size),
             "[FoundationposeRender] cudaMalloc `mesh_faces_device` FAILED!!!");
  vertices_device_ = DeviceBufferUniquePtrType<float>(_vertices_device, CudaMemoryDeleter<float>());

  CHECK_CUDA(cudaMalloc(&_vertex_normals_device, vertices_size),
             "[FoundationposeRender] cudaMalloc `vertex_normals_device` FAILED!!!");
  vertex_normals_device_ =
      DeviceBufferUniquePtrType<float>(_vertex_normals_device, CudaMemoryDeleter<float>());

  CHECK_CUDA(cudaMalloc(&_mesh_faces_device, faces_size),
             "[FoundationposeRender] cudaMalloc `mesh_faces_device` FAILED!!!");
  mesh_faces_device_ =
      DeviceBufferUniquePtrType<int32_t>(_mesh_faces_device, CudaMemoryDeleter<int32_t>());

  CHECK_CUDA(cudaMalloc(&_texcoords_device, texcoords_size),
             "[FoundationposeRender] cudaMalloc `texcoords_device_` FAILED!!!");
  texcoords_device_ =
      DeviceBufferUniquePtrType<float>(_texcoords_device, CudaMemoryDeleter<float>());

  CHECK_CUDA(cudaMalloc(&_texture_map_device, rgb_texture_map.total() * kNumChannels),
             "[FoundationposeRender] cudaMalloc `texture_map_device_` FAILED!!!");
  texture_map_device_ =
      DeviceBufferUniquePtrType<uint8_t>(_texture_map_device, CudaMemoryDeleter<uint8_t>());

  CHECK_CUDA(
      cudaMemcpy(vertices_device_.get(), vertices_.data(), vertices_size, cudaMemcpyHostToDevice),
      "[FoundationposeRender] cudaMemcpy mesh_faces_host -> mesh_faces_device FAILED!!!");
  CHECK_CUDA(cudaMemcpy(vertex_normals_device_.get(), vertex_normals.data(), vertices_size,
                        cudaMemcpyHostToDevice),
             "[FoundationposeRender] cudaMemcpy mesh_faces_host -> mesh_faces_device FAILED!!!");
  CHECK_CUDA(
      cudaMemcpy(mesh_faces_device_.get(), mesh_faces_.data(), faces_size, cudaMemcpyHostToDevice),
      "[FoundationposeRender] cudaMemcpy mesh_faces_host -> mesh_faces_device FAILED!!!");
  CHECK_CUDA(cudaMemcpy(texcoords_device_.get(), texcoords_.data(),
                        texcoords_.size() * sizeof(float), cudaMemcpyHostToDevice),
             "[FoundationposeRender] cudaMemcpy texcoords_host -> texcoords_device_ FAILED!!!");
  CHECK_CUDA(
      cudaMemcpy(texture_map_device_.get(), reinterpret_cast<uint8_t *>(rgb_texture_map.data),
                 rgb_texture_map.total() * kNumChannels, cudaMemcpyHostToDevice),
      "[FoundationposeRender] cudaMemcpy rgb_texture_map_host -> texture_map_device_ FAILED!!!");

  // Preprocess mesh data
  nvcv::Tensor texture_map_tensor;
  WrapImgPtrToNHWCTensor(texture_map_device_.get(), texture_map_tensor, 1, texture_map_height_,
                         texture_map_width_, kNumChannels);

  nvcv::TensorShape::ShapeType shape{1, texture_map_height_, texture_map_width_, kNumChannels};
  nvcv::TensorShape            tensor_shape{shape, "NHWC"};
  float_texture_map_tensor_ = nvcv::Tensor(tensor_shape, nvcv::TYPE_F32);

  const float       scale_factor = 1.0f / 255.0f;
  cvcuda::ConvertTo convert_op;
  convert_op(cuda_stream_render_, texture_map_tensor, float_texture_map_tensor_, scale_factor,
             0.0f);

  return true;
}

bool FoundationPoseRenderer::TransformVerticesOnCUDA(cudaStream_t                        stream,
                                                     const std::vector<Eigen::MatrixXf> &tfs,
                                                     float *output_buffer)
{
  // Get the dimensions of the inputs
  int tfs_size = tfs.size();
  CHECK_STATE(tfs_size != 0, "[FoundationposeRender] The transfomation matrix is empty! ");

  CHECK_STATE(tfs[0].cols() == tfs[0].rows(),
              "[FoundationposeRender] The transfomation matrix has different rows and cols! ");

  const int total_elements = tfs[0].cols() * tfs[0].rows();

  float *transform_device_buffer_ = nullptr;
  cudaMallocAsync(&transform_device_buffer_, tfs_size * total_elements * sizeof(float), stream);

  for (int i = 0; i < tfs_size; ++i)
  {
    cudaMemcpyAsync(transform_device_buffer_ + i * total_elements, tfs[i].data(),
                    total_elements * sizeof(float), cudaMemcpyHostToDevice, stream);
  }

  foundationpose_render::transform_points(stream, transform_device_buffer_, tfs_size,
                                          vertices_device_.get(), num_vertices_, output_buffer);

  cudaFreeAsync(transform_device_buffer_, stream);
  return true;
}

bool FoundationPoseRenderer::TransformVertexNormalsOnCUDA(cudaStream_t stream,
                                                          const std::vector<Eigen::MatrixXf> &tfs,
                                                          float *output_buffer)
{
  // Get the dimensions of the inputs
  int tfs_size = tfs.size();
  CHECK_STATE(tfs_size != 0, "[FoundationposeRender] The transfomation matrix is empty! ");

  CHECK_STATE(tfs[0].cols() == tfs[0].rows(),
              "[FoundationposeRender] The transfomation matrix has different rows and cols! ");

  const int total_elements = tfs[0].cols() * tfs[0].rows();

  float *transform_device_buffer_ = nullptr;
  cudaMallocAsync(&transform_device_buffer_, tfs_size * total_elements * sizeof(float), stream);

  for (int i = 0; i < tfs_size; ++i)
  {
    cudaMemcpyAsync(transform_device_buffer_ + i * total_elements, tfs[i].data(),
                    total_elements * sizeof(float), cudaMemcpyHostToDevice, stream);
  }

  foundationpose_render::transform_normals(stream, transform_device_buffer_, tfs_size,
                                           vertex_normals_device_.get(), num_vertices_,
                                           output_buffer);

  cudaFreeAsync(transform_device_buffer_, stream);
  return true;
}

bool FoundationPoseRenderer::GeneratePoseClipOnCUDA(cudaStream_t stream,
                                                    float       *output_buffer,
                                                    const std::vector<Eigen::MatrixXf> &poses,
                                                    const RowMajorMatrix               &bbox2d,
                                                    const Eigen::Matrix3f              &K,
                                                    int                                 rgb_H,
                                                    int                                 rgb_W)
{
  const int tfs_size = poses.size();
  CHECK_STATE(tfs_size > 0, "[FoundationPoseRender] `GeneratePoseClip` Got empty poses!!!");
  Eigen::Matrix4f projection_mat;
  CHECK_STATE(ProjectMatrixFromIntrinsics(projection_mat, K, rgb_H, rgb_W),
              "[FoundationPoseRender] ProjectMatrixFromIntrinsics Failed!!!");

  float    *transform_buffer_device;
  const int transform_total_elements = poses[0].cols() * poses[0].rows();
  cudaMallocAsync(&transform_buffer_device, tfs_size * transform_total_elements * sizeof(float),
                  stream);
  for (int i = 0; i < tfs_size; ++i)
  {
    Eigen::Matrix4f transform_matrix = projection_mat * (kGLCamInCVCam * poses[i]);
    cudaMemcpyAsync(transform_buffer_device + i * transform_total_elements, transform_matrix.data(),
                    transform_total_elements * sizeof(float), cudaMemcpyHostToDevice, stream);
  }

  float    *bbox2d_buffer_device;
  const int bbox2d_matrix_total_elements = bbox2d.rows() * bbox2d.cols();
  cudaMallocAsync(&bbox2d_buffer_device, bbox2d_matrix_total_elements * sizeof(float), stream);
  cudaMemcpyAsync(bbox2d_buffer_device, bbox2d.data(), bbox2d_matrix_total_elements * sizeof(float),
                  cudaMemcpyHostToDevice, stream);

  foundationpose_render::generate_pose_clip(stream, transform_buffer_device, bbox2d_buffer_device,
                                            tfs_size, vertices_device_.get(), num_vertices_,
                                            output_buffer, rgb_H, rgb_W);

  cudaFreeAsync(bbox2d_buffer_device, stream);
  cudaFreeAsync(transform_buffer_device, stream);

  return true;
}

bool FoundationPoseRenderer::NvdiffrastRender(cudaStream_t                        cuda_stream,
                                              const std::vector<Eigen::MatrixXf> &poses,
                                              const Eigen::Matrix3f              &K,
                                              const RowMajorMatrix               &bbox2d,
                                              int                                 rgb_H,
                                              int                                 rgb_W,
                                              int                                 H,
                                              int                                 W,
                                              nvcv::Tensor                       &flip_color_tensor,
                                              nvcv::Tensor &flip_xyz_map_tensor)
{
  size_t N = poses.size();
  CHECK_STATE(TransformVerticesOnCUDA(cuda_stream, poses, pts_cam_device_.get()),
              "[FoundationPoseRender] Failed transform mesh vertices points !!!");

  CHECK_STATE(
      GeneratePoseClipOnCUDA(cuda_stream, pose_clip_device_.get(), poses, bbox2d, K, rgb_H, rgb_W),
      "[FoundationposeRender] `GeneratePoseClipCUDA` Failed !!!");

  foundationpose_render::rasterize(cuda_stream, cr_.get(), pose_clip_device_.get(),
                                   mesh_faces_device_.get(), rast_out_device_.get(), num_vertices_,
                                   num_faces_, H, W, N);
  CHECK_CUDA(cudaGetLastError(), "[FoundationPoseRenderer] rasterize failed!!!");

  foundationpose_render::interpolate(cuda_stream, pts_cam_device_.get(), rast_out_device_.get(),
                                     mesh_faces_device_.get(), xyz_map_device_.get(), num_vertices_,
                                     num_faces_, 3, kVertexPoints, H, W, N);
  CHECK_CUDA(cudaGetLastError(), "[FoundationPoseRenderer] interpolate failed!!!");

  foundationpose_render::interpolate(cuda_stream, texcoords_device_.get(), rast_out_device_.get(),
                                     mesh_faces_device_.get(), texcoords_out_device_.get(),
                                     num_vertices_, num_faces_, 2, kTexcoordsDim, H, W, N);
  CHECK_CUDA(cudaGetLastError(), "[FoundationPoseRenderer] interpolate failed!!!");

  auto float_texture_map_data = float_texture_map_tensor_.exportData<nvcv::TensorDataStridedCuda>();
  foundationpose_render::texture(cuda_stream,
                                 reinterpret_cast<float *>(float_texture_map_data->basePtr()),
                                 texcoords_out_device_.get(), color_device_.get(),
                                 texture_map_height_, texture_map_width_, kNumChannels, 1, H, W, N);
  CHECK_CUDA(cudaGetLastError(), "[FoundationPoseRenderer] texture failed!!!");

  CHECK_STATE(TransformVertexNormalsOnCUDA(cuda_stream, poses, diffuse_intensity_device_.get()),
              "[FoundationPoseRenderer] Transform vertex normals failed!!!");

  foundationpose_render::interpolate(cuda_stream, diffuse_intensity_device_.get(),
                                     rast_out_device_.get(), mesh_faces_device_.get(),
                                     diffuse_intensity_map_device_.get(), num_vertices_, num_faces_,
                                     3, 1, H, W, N);
  CHECK_CUDA(cudaGetLastError(), "[FoundationPoseRenderer] interpolate failed!!!");

  foundationpose_render::refine_color(cuda_stream, color_device_.get(),
                                      diffuse_intensity_map_device_.get(), rast_out_device_.get(),
                                      color_device_.get(), poses.size(), 0.8, 0.5, H, W);
  CHECK_CUDA(cudaGetLastError(), "[FoundationPoseRenderer] refine_color failed!!!");

  float min_value = 0.0;
  float max_value = 1.0;
  foundationpose_render::clamp(cuda_stream, color_device_.get(), min_value, max_value,
                               N * H * W * kNumChannels);
  CHECK_CUDA(cudaGetLastError(), "[FoundationPoseRenderer] clamp failed!!!");

  nvcv::Tensor color_tensor, xyz_map_tensor;
  WrapFloatPtrToNHWCTensor(color_device_.get(), color_tensor, N, H, W, kNumChannels);
  WrapFloatPtrToNHWCTensor(xyz_map_device_.get(), xyz_map_tensor, N, H, W, kNumChannels);

  cvcuda::Flip flip_op;
  flip_op(cuda_stream, color_tensor, flip_color_tensor, 0);
  CHECK_CUDA(cudaGetLastError(), "[FoundationPoseRenderer] flip_op failed!!!");

  flip_op(cuda_stream, xyz_map_tensor, flip_xyz_map_tensor, 0);
  CHECK_CUDA(cudaGetLastError(), "[FoundationPoseRenderer] flip_op failed!!!");
  return true;
}

bool FoundationPoseRenderer::RenderProcess(cudaStream_t                        cuda_stream,
                                           const std::vector<Eigen::MatrixXf> &poses,
                                           const std::vector<RowMajorMatrix>  &tfs,
                                           void                               *poses_on_device,
                                           int                                 input_image_height,
                                           int                                 input_image_width,
                                           void                               *render_input_dst_ptr)
{
  const int N = poses.size();
  // Convert the bbox2d from vector N of 2*2 matrix into a N*4 matrix
  RowMajorMatrix bbox2d(tfs.size(), 4);
  CHECK_STATE(ConstructBBox2D(bbox2d, tfs, crop_window_H_, crop_window_W_),
              "[FoundationPose Render] RenderProcess construct bbox2d failed!!!");

  // render
  nvcv::Tensor render_rgb_tensor;
  nvcv::Tensor render_xyz_map_tensor;
  WrapFloatPtrToNHWCTensor(render_crop_rgb_tensor_device_.get(), render_rgb_tensor, N,
                           crop_window_H_, crop_window_W_, kNumChannels);
  WrapFloatPtrToNHWCTensor(render_crop_xyz_map_tensor_device_.get(), render_xyz_map_tensor, N,
                           crop_window_H_, crop_window_W_, kNumChannels);

  // Render the object using give poses
  CHECK_STATE(NvdiffrastRender(cuda_stream, poses, intrinsic_, bbox2d, input_image_height,
                               input_image_width, crop_window_H_, crop_window_W_, render_rgb_tensor,
                               render_xyz_map_tensor),
              "[FoundationPose Render] RenderProcess NvdiffrastRender failed!!!");

  auto render_rgb_data     = render_rgb_tensor.exportData<nvcv::TensorDataStridedCuda>();
  auto render_xyz_map_data = render_xyz_map_tensor.exportData<nvcv::TensorDataStridedCuda>();

  foundationpose_render::threshold_and_downscale_pointcloud(
      cuda_stream, reinterpret_cast<float *>(render_xyz_map_data->basePtr()),
      reinterpret_cast<float *>(poses_on_device), N, crop_window_H_ * crop_window_W_,
      mesh_diameter_ / 2, min_depth_, max_depth_);
  CHECK_CUDA(cudaGetLastError(), "[FoundationPose] RenderProcess threshold_and... FAILED!!!");

  foundationpose_render::concat(cuda_stream, reinterpret_cast<float *>(render_rgb_data->basePtr()),
                                reinterpret_cast<float *>(render_xyz_map_data->basePtr()),
                                reinterpret_cast<float *>(render_input_dst_ptr), N, crop_window_H_,
                                crop_window_W_, kNumChannels, kNumChannels);
  CHECK_CUDA(cudaGetLastError(), "[FoundationPose] RenderProcess concat FAILED!!!");

  return true;
}

bool FoundationPoseRenderer::TransfProcess(cudaStream_t                       cuda_stream,
                                           void                              *rgb_on_device,
                                           void                              *xyz_map_on_device,
                                           int                                input_image_height,
                                           int                                input_image_width,
                                           const std::vector<RowMajorMatrix> &tfs,
                                           void                              *poses_on_device,
                                           void                              *transf_input_dst_ptr)
{
  // crop rgb (transformed)
  const size_t N = tfs.size();

  nvcv::Tensor rgb_tensor;
  nvcv::Tensor xyz_map_tensor;
  WrapImgPtrToNHWCTensor(reinterpret_cast<uint8_t *>(rgb_on_device), rgb_tensor, 1,
                         input_image_height, input_image_width, kNumChannels);

  WrapFloatPtrToNHWCTensor(reinterpret_cast<float *>(xyz_map_on_device), xyz_map_tensor, 1,
                           input_image_height, input_image_width, kNumChannels);

  const int    rgb_flags    = NVCV_INTERP_LINEAR;
  const int    xyz_flags    = NVCV_INTERP_NEAREST;
  const float4 border_value = {0, 0, 0, 0};

  const float             scale_factor = 1.0f / 255.0f;
  cvcuda::WarpPerspective warpPerspectiveOp(0);
  cvcuda::ConvertTo       convert_op;
  nvcv::Tensor            transformed_rgb_tensor;
  WrapImgPtrToNHWCTensor(transformed_crop_rgb_tensor_device_.get(), transformed_rgb_tensor, 1,
                         crop_window_H_, crop_window_W_, kNumChannels);

  for (size_t index = 0; index < N; index++)
  {
    nvcv::Tensor float_rgb_tensor;
    nvcv::Tensor transformed_xyz_map_tensor;

    // get ptr offset from index
    const size_t single_batch_element_size = crop_window_H_ * crop_window_W_ * kNumChannels;
    WrapFloatPtrToNHWCTensor(transformed_rgb_device_.get() + index * single_batch_element_size,
                             float_rgb_tensor, 1, crop_window_H_, crop_window_W_, kNumChannels);

    WrapFloatPtrToNHWCTensor(transformed_xyz_map_device_.get() + index * single_batch_element_size,
                             transformed_xyz_map_tensor, 1, crop_window_H_, crop_window_W_,
                             kNumChannels);

    NVCVPerspectiveTransform trans_matrix;
    for (size_t i = 0; i < kPTMatrixDim; i++)
    {
      for (size_t j = 0; j < kPTMatrixDim; j++)
      {
        trans_matrix[i * kPTMatrixDim + j] = tfs[index](i, j);
      }
    }

    warpPerspectiveOp(cuda_stream, rgb_tensor, transformed_rgb_tensor, trans_matrix, rgb_flags,
                      NVCV_BORDER_CONSTANT, border_value);
    CHECK_CUDA(cudaGetLastError(),
               "[FoundationPose] TransfProcess warpPerspectiveOp on rgb FAILED!!!");

    convert_op(cuda_stream, transformed_rgb_tensor, float_rgb_tensor, scale_factor, 0.0f);
    CHECK_CUDA(cudaGetLastError(), "[FoundationPose] TransfProcess convert_op on rgb FAILED!!!");

    warpPerspectiveOp(cuda_stream, xyz_map_tensor, transformed_xyz_map_tensor, trans_matrix,
                      xyz_flags, NVCV_BORDER_CONSTANT, border_value);
    CHECK_CUDA(cudaGetLastError(),
               "[FoundationPose] TransfProcess warpPerspectiveOp on xyz_map FAILED!!!");
  }

  foundationpose_render::threshold_and_downscale_pointcloud(
      cuda_stream, transformed_xyz_map_device_.get(), reinterpret_cast<float *>(poses_on_device), N,
      crop_window_W_ * crop_window_H_, mesh_diameter_ / 2, min_depth_, max_depth_);
  CHECK_CUDA(cudaGetLastError(), "[FoundationPose] TransfProcess threshold_and... FAILED!!!");

  // concat 到缓存上
  foundationpose_render::concat(cuda_stream, transformed_rgb_device_.get(),
                                transformed_xyz_map_device_.get(),
                                reinterpret_cast<float *>(transf_input_dst_ptr), N, crop_window_H_,
                                crop_window_W_, kNumChannels, kNumChannels);
  CHECK_CUDA(cudaGetLastError(), "[FoundationPose] TransfProcess concat FAILED!!!");

  return true;
}

bool FoundationPoseRenderer::RenderAndTransform(const std::vector<Eigen::Matrix4f> &_poses,
                                                void                               *rgb_on_device,
                                                void                               *depth_on_device,
                                                void *xyz_map_on_device,
                                                int   input_image_height,
                                                int   input_image_width,
                                                void *render_buffer,
                                                void *transf_buffer,
                                                float crop_ratio)
{
  const int input_poses_num = _poses.size();

  // 1. 根据目标位姿计算变换矩阵
  std::vector<Eigen::MatrixXf> poses(_poses.begin(), _poses.end());
  Eigen::Vector2i              out_size = {crop_window_H_, crop_window_W_};
  auto tfs = ComputeCropWindowTF(poses, intrinsic_, out_size, crop_ratio, mesh_diameter_);
  CHECK_STATE(tfs.size() != 0, "[FoundationposeRender] The transform matrix vector is empty");

  // 2. 将输入的poses拷贝到device端
  float *_poses_device = static_cast<float *>(input_poses_device_.get());
  for (size_t i = 0; i < poses.size(); ++i)
  {
    CHECK_CUDA(cudaMemcpy(&_poses_device[i * 16], _poses[i].data(), 16 * sizeof(float),
                          cudaMemcpyHostToDevice),
               "[FoundationposeRender] cudaMemcpy poses_host -> poses_device FAILED!!!");
  }

  // 3. 根据poses和tfs裁剪输入rgb和xyz_map，并Transpose后填充至目标缓存
  CHECK_STATE(
      TransfProcess(cuda_stream_transf_, rgb_on_device, xyz_map_on_device, input_image_height,
                    input_image_width, tfs, input_poses_device_.get(), transf_buffer),
      "[FoundationPose Renderer] TransfProcess Failed!!!");
  // 4. 根据poses和tfs渲染rgb图像和xyz_map，并Transpose后填充至目标缓存
  CHECK_STATE(RenderProcess(cuda_stream_render_, poses, tfs, input_poses_device_.get(),
                            input_image_height, input_image_width, render_buffer),
              "[FoundationPose Renderer] RenderProcess Failed!!!");

  // 同步render和transform流程的cuda_stream，确保退出前任务全部完成
  CHECK_CUDA(cudaStreamSynchronize(cuda_stream_transf_),
             "[FoundationPose Renderer] cudaStreamSync `cuda_stream_transf_` FAILED!!!");
  CHECK_CUDA(cudaStreamSynchronize(cuda_stream_render_),
             "[FoundationPose Renderer] cudaStreamSync `cuda_stream_render` FAILED!!!");
  return true;
}

} // namespace detection_6d
