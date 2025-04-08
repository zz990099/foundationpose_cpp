#pragma once

#include <vector>
#include <Eigen/Dense>
#include <opencv2/core.hpp>

namespace detection_6d {

/**
 * @brief Abstract base class for mesh loading and data access interfaces
 *
 * @note Implementations should handle different mesh formats while maintaining
 *       consistent vertex/face data organization
 */
class BaseMeshLoader {
public:
  virtual ~BaseMeshLoader() = default;

  using Vector3ui = Eigen::Matrix<uint32_t, 3, 1>;

  /**
   * @brief Get identifier name for the loaded mesh
   * @return Mesh name string
   */
  virtual std::string GetName() const noexcept = 0;

  /** @brief Get diameter of the mesh's bounding sphere */
  virtual float GetMeshDiameter() const noexcept = 0;

  /** @brief Get number of vertices in the mesh */
  virtual size_t GetMeshNumVertices() const noexcept = 0;

  /** @brief Get number of triangular faces in the mesh */
  virtual size_t GetMeshNumFaces() const noexcept = 0;

  /** @brief Access array of vertex positions (3D coordinates) */
  virtual const std::vector<Eigen::Vector3f> &GetMeshVertices() const noexcept = 0;

  /** @brief Access array of vertex normals */
  virtual const std::vector<Eigen::Vector3f> &GetMeshVertexNormals() const noexcept = 0;

  /** @brief Access array of texture coordinates (UV mapping) */
  virtual const std::vector<Eigen::Vector3f> &GetMeshTextureCoords() const noexcept = 0;

  /** @brief Access array of triangular face indices */
  virtual const std::vector<Vector3ui> &GetMeshTriangleFaces() const noexcept = 0;

  /** @brief Get centroid position of the mesh model */
  virtual const Eigen::Vector3f &GetMeshModelCenter() const noexcept = 0;

  /**
   * @brief Get orientation bounds transformation matrix
   * @return 4x4 matrix containing axis-aligned bounding box orientation
   */
  virtual const Eigen::Matrix4f &GetOrientBounds() const noexcept = 0;

  /** @brief Get dimensional measurements of the object (width/height/depth) */
  virtual const Eigen::Vector3f &GetObjectDimension() const noexcept = 0;

  /** @brief Access texture map image for the mesh */
  virtual const cv::Mat &GetTextureMap() const noexcept = 0;
};

/**
 * @brief Convert pose from mesh coordinate frame to bounding box frame
 *
 * @param pose_in_mesh Input pose in mesh coordinate system
 * @param mesh_loader Mesh loader containing transformation parameters
 * @return Transformed pose in bounding box coordinate system
 *
 * @note Transformation formula:
 *       T_bbox = T_mesh * T_center * T_orient
 *       Where T_center translates to mesh center, T_orient aligns with bounds
 */
inline Eigen::Matrix4f ConvertPoseMesh2BBox(const Eigen::Matrix4f                 &pose_in_mesh,
                                            const std::shared_ptr<BaseMeshLoader> &mesh_loader)
{
  Eigen::Matrix4f tf_to_center   = Eigen::Matrix4f::Identity();
  tf_to_center.block<3, 1>(0, 3) = -mesh_loader->GetMeshModelCenter();
  return pose_in_mesh * tf_to_center * mesh_loader->GetOrientBounds();
}

/**
 * @brief Factory function for creating Assimp-based mesh loader
 *
 * @param name Identifier for the mesh
 * @param mesh_file_path Path to mesh file (supports .obj/.ply/.stl etc.)
 * @return Shared pointer to initialized mesh loader instance
 *
 * @note Throws std::runtime_error if mesh loading fails
 */
std::shared_ptr<BaseMeshLoader> CreateAssimpMeshLoader(const std::string &name,
                                                       const std::string &mesh_file_path);

} // namespace detection_6d
