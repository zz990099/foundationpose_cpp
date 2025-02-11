// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

#include "foundationpose_utils.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

#include "cuda.h"
#include "cuda_runtime.h"

namespace detection_6d {

std::pair<Eigen::Vector3f, Eigen::Vector3f> FindMinMaxVertex(const aiMesh* mesh) {
  Eigen::Vector3f min_vertex = {0, 0, 0};
  Eigen::Vector3f max_vertex = {0, 0, 0};

  if (mesh->mNumVertices == 0) {
    return std::pair{min_vertex, max_vertex};
  }

  min_vertex << mesh->mVertices[0].x, mesh->mVertices[0].y, mesh->mVertices[0].z;
  max_vertex << mesh->mVertices[0].x, mesh->mVertices[0].y, mesh->mVertices[0].z;

  // Iterate over all vertices to find the bounding box
  for (size_t v = 0; v < mesh->mNumVertices; v++) {
    float vx = mesh->mVertices[v].x;
    float vy = mesh->mVertices[v].y;
    float vz = mesh->mVertices[v].z;

    min_vertex[0] = std::min(min_vertex[0], vx);
    min_vertex[1] = std::min(min_vertex[1], vy);
    min_vertex[2] = std::min(min_vertex[2], vz);

    max_vertex[0] = std::max(max_vertex[0], vx);
    max_vertex[1] = std::max(max_vertex[1], vy);
    max_vertex[2] = std::max(max_vertex[2], vz);
  }
  return std::pair{min_vertex, max_vertex};
}

float CalcMeshDiameter(const aiMesh* mesh) {
  float max_dist = 0.0;
  for (unsigned int i = 0; i < mesh->mNumVertices; ++i) {
    for (unsigned int j = i + 1; j < mesh->mNumVertices; ++j) {
      aiVector3D diff = mesh->mVertices[i] - mesh->mVertices[j];
      float dist = diff.Length();
      max_dist = std::max(max_dist, dist);
    }
  }
  return max_dist;
}



void ComputeOBB(const aiMesh* mesh, 
                Eigen::Matrix4f& out_orient_bbox, 
                Eigen::Vector3f& out_dimension) 
{
  std::vector<Eigen::Vector3f> vertices;
  for (unsigned int i = 0; i < mesh->mNumVertices; ++i) {
      vertices.emplace_back(mesh->mVertices[i].x, mesh->mVertices[i].y, mesh->mVertices[i].z);
  }

  // 计算质心
  Eigen::Vector3f mean = Eigen::Vector3f::Zero();
  for (const auto& v : vertices) {
      mean += v;
  }
  mean /= vertices.size();

  // 计算协方差矩阵
  Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();
  for (const auto& v : vertices) {
      Eigen::Vector3f diff = v - mean;
      cov += diff * diff.transpose();
  }
  cov /= vertices.size();

  // 特征分解
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(cov);
  Eigen::Matrix3f rotation = solver.eigenvectors();
  Eigen::Vector3f extents = solver.eigenvalues().cwiseSqrt();
  // 生成变换矩阵
  Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
  transform.block<3, 3>(0, 0) = rotation;
  transform.block<3, 1>(0, 3) = mean;

  Eigen::MatrixXf transformed(vertices.size(), 3);
  for (int i = 0 ; i < vertices.size() ; ++ i) {
    Eigen::Vector3f proj = rotation.transpose() * vertices[i];
    transformed(i, 0) = proj(0);
    transformed(i, 1) = proj(1);
    transformed(i, 2) = proj(2);
  }

  Eigen::Vector3f minBound = transformed.colwise().minCoeff();
  Eigen::Vector3f maxBound = transformed.colwise().maxCoeff();

  Eigen::Vector3f dimension = maxBound - minBound;

  out_orient_bbox = transform;
  out_dimension = dimension;
}


TexturedMeshLoader::TexturedMeshLoader(const std::string& mesh_file_path,
                                      const std::string& textured_file_path)
{
  // 1. load textured mesh file using assimp
  if (mesh_file_path.empty() || textured_file_path.empty()) {
    throw std::invalid_argument("[TexturedMeshLoader] Invalid textured mesh file path: "
                              + mesh_file_path + "\t" + textured_file_path);
  }
  LOG(INFO) << "Loading mesh file: " << mesh_file_path;
  Assimp::Importer importer;
  const aiScene* scene = importer.ReadFile(
    mesh_file_path,
    aiProcess_Triangulate | aiProcess_JoinIdenticalVertices | aiProcess_SortByPType
  );
  if (scene == nullptr) {
    throw std::runtime_error("[TexturedMeshLoader] Failed to read mesh file: "
                            + mesh_file_path);
  }

  const aiMesh* mesh = scene->mMeshes[0];
  mesh_diamter_ = CalcMeshDiameter(mesh);
  ComputeOBB(mesh, obb_, dim_);

  auto min_max_vertex = FindMinMaxVertex(mesh);
  mesh_center_ = (min_max_vertex.second + min_max_vertex.first) / 2.0;

  // Walk through each of the mesh's vertices
  for (unsigned int v = 0; v < mesh->mNumVertices; v++) {
    vertices_.push_back(mesh->mVertices[v]);
  }
  for (unsigned int i = 0 ; i < AI_MAX_NUMBER_OF_TEXTURECOORDS ; ++ i) {
    if (mesh->mTextureCoords[i] != nullptr) {
      std::vector<aiVector3D> tex_coords_vec;
      tex_coords_vec.reserve(mesh->mNumVertices);
      for (int v = 0 ; v < mesh->mNumVertices ; ++ v) {
        tex_coords_vec[v] = mesh->mTextureCoords[i][v];
      }
      texcoords_.push_back(std::move(tex_coords_vec));
    }
  }

  for (unsigned int f = 0 ; f < mesh->mNumFaces ; ++ f){ 
    faces_.push_back(mesh->mFaces[f]);
  }

  LOG(INFO) << "Loading textured map file: " << textured_file_path;
  texture_map_ = cv::imread(textured_file_path);
  if (texture_map_.empty()) {
    throw std::runtime_error("[TexturedMeshLoader] Failed to read textured image: "
                            + textured_file_path);
  }
  cv::cvtColor(texture_map_, texture_map_, cv::COLOR_BGR2RGB);


  LOG(INFO) << "Successfully Loaded textured mesh file!!!";
  LOG(INFO) << "Mesh has vertices_num: " << vertices_.size()
            << ", diameter: " << mesh_diamter_
            << ", faces_num: " << faces_.size()
            << ", center: " << mesh_center_;
}





/**
 * @brief 获取mesh模型的半径
 * 
 * @return float 
 */
float 
TexturedMeshLoader::GetMeshDiameter() const noexcept
{
  return mesh_diamter_;
}

/**
 * @brief 获取mesh模型的顶点数量
 * 
 * @return size_t 
 */
size_t 
TexturedMeshLoader::GetMeshNumVertices() const noexcept
{
  return vertices_.size();
}

/**
 * @brief 获取mesh模型的顶点数据指针
 * 
 * @return const std::vector<aiVector3D> &
 */
const std::vector<aiVector3D> &
TexturedMeshLoader::GetMeshVertices() const noexcept
{
  return vertices_;
}

/**
 * @brief 获取mesh模型的外观坐标系
 * 
 * @return const std::vector<aiVector3D> &
 */
const std::vector<std::vector<aiVector3D>> &
TexturedMeshLoader::GetMeshTextureCoords() const noexcept
{
  return texcoords_;
}

/**
 * @brief 获取mesh模型的faces
 * 
 * @return const std::vector<aiFace> &
 */
const std::vector<aiFace> &
TexturedMeshLoader::GetMeshFaces() const noexcept
{
  return faces_;
}


  /**
 * @brief 获取mesh模型的三维中心
 * 
 * @return const std::vector<Eigen::Vector3f>& 
 */
const Eigen::Vector3f&
TexturedMeshLoader::GetMeshModelCenter() const noexcept
{
  return mesh_center_;
}


  /**
 * @brief 获取mesh包围盒转换矩阵
 * 
 * @return const Eigen::Matrix4f& 
 */
const Eigen::Matrix4f& 
TexturedMeshLoader::GetOrientBounds() const noexcept
{
  return obb_;
}



/**
 * @brief 获取cv::Mat格式的外观图
 * 
 * @return const cv::Mat& 
 */
const cv::Mat& 
TexturedMeshLoader::GetTextureMap() const noexcept
{
  return texture_map_;
}




/**
 * @brief 获取物体最小包络盒的尺寸
 * 
 * @return const Eigen::Vector3f 
 */
const Eigen::Vector3f 
TexturedMeshLoader::GetObjectDimension() const noexcept
{
  return dim_;  
}


} // namespace detection_6d