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

#ifndef NVIDIA_ISAAC_ROS_EXTENSIONS_FOUNDATIONPOSE_UTILS_HPP_
#define NVIDIA_ISAAC_ROS_EXTENSIONS_FOUNDATIONPOSE_UTILS_HPP_

#include <iostream>

#include <Eigen/Dense>

#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <assimp/Importer.hpp>
#include <opencv2/core.hpp>

#include "cuda.h"
#include "cuda_runtime.h"

#include "deploy_core/async_pipeline.h"

namespace detection_6d {

struct FoundationPosePipelinePackage : public async_pipeline::IPipelinePackage {
  // 输入host端rgb图像
  cv::Mat rgb_on_host;
  // 输入host端depth
  cv::Mat depth_on_host;
  // 输入host端mask
  cv::Mat mask_on_host;
  // 目标物名称
  std::string target_name;

  int input_image_height;

  int input_image_width;

  // device端的输入图像缓存
  std::shared_ptr<void> rgb_on_device;
  // device端的输入深度缓存
  std::shared_ptr<void> depth_on_device;
  // device端由depth转换得到的xyz_map
  std::shared_ptr<void> xyz_map_on_device;
  // 生成的假设位姿
  std::vector<Eigen::Matrix4f> hyp_poses;

  // 保存refine阶段用的推理缓存
  std::shared_ptr<inference_core::IBlobsBuffer> refiner_blobs_buffer;
  // 保存score阶段用的推理缓存
  std::shared_ptr<inference_core::IBlobsBuffer> scorer_blobs_buffer;
  // 保存用于推理的blob_buffer
  std::shared_ptr<inference_core::IBlobsBuffer> infer_buffer;

  // **最终输出的位姿** //
  Eigen::Matrix4f actual_pose;

  std::shared_ptr<inference_core::IBlobsBuffer> GetInferBuffer()
  {
    return infer_buffer;
  }
};

#define CHECK_CUDA(result, hint)                    \
  {                                                 \
    auto res = (result);                            \
    if (res != cudaSuccess)                         \
    {                                               \
      LOG(ERROR) << hint << "  CudaError: " << res; \
      return false;                                 \
    }                                               \
  }

#define MESSURE_DURATION_AND_CHECK_CUDA(run, hint)                                           \
  {                                                                                          \
    auto start = std::chrono::high_resolution_clock::now();                                  \
    CHECK_CUDA((run), hint);                                                                 \
    auto end = std::chrono::high_resolution_clock::now();                                    \
    LOG(INFO) << #run << " cost(us): "                                                       \
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count(); \
  }

#define CHECK_CUDA_THROW(result, hint) \
  {                                    \
    if ((result) != cudaSuccess)       \
    {                                  \
      LOG(ERROR) << hint;              \
      throw std::runtime_error(hint);  \
    }                                  \
  }

template <typename T>
class CudaMemoryDeleter {
public:
  void operator()(T *ptr)
  {
    auto suc = cudaFree(ptr);
    if (suc != cudaSuccess)
    {
      LOG(INFO) << "Release cuda memory ptr FAILED!!!";
    }
  }
};

} // namespace detection_6d

#endif // NVIDIA_ISAAC_ROS_EXTENSIONS_FOUNDATIONPOSE_UTILS_HPP_
