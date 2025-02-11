/*
 * @Description:
 * @Author: Teddywesside 18852056629@163.com
 * @Date: 2024-11-25 14:00:38
 * @LastEditTime: 2024-11-26 09:31:01
 * @FilePath: /EasyDeploy/inference_core/trt_core/src/trt_blob_buffer.hpp
 */
#ifndef __EASY_DEPLOY_TRT_BLOB_BUFFER_H
#define __EASY_DEPLOY_TRT_BLOB_BUFFER_H

#include <cuda_runtime.h>

#include "deploy_core/blob_buffer.h"

namespace inference_core {

template <typename Type>
inline Type CumVector(const std::vector<Type> &vec)
{
  Type ret = 1;
  for (const auto &nn : vec)
  {
    ret *= nn;
  }

  return ret;
}

template <typename Type>
inline std::string VisualVec(const std::vector<Type> &vec)
{
  std::string ret;
  for (const auto &v : vec)
  {
    ret += std::to_string(v) + " ";
  }
  return ret;
}

class TrtBlobBuffer : public IBlobsBuffer {
public:
  /**
   * @brief Overrided from `IBlobsBuffer`, provide the buffer ptr which is used as
   * input data of tensorrt inference engine. It depends on `SetBlobBuffer` method.
   *
   * @param blob_name The blob_name of model.
   * @return std::pair<void*, DataLocation> . Will return {nullptr, UNKOWN} if `blob_name`
   * does not match.
   */
  std::pair<void *, DataLocation> GetOuterBlobBuffer(const std::string &blob_name) noexcept override
  {
    if (outer_map_blob2ptr_.find(blob_name) == outer_map_blob2ptr_.end())
    {
      LOG(ERROR) << "[TrtBlobBuffer] `GetOuterBlobBuffer` Got invalid `blob_name`: " << blob_name;
      return {nullptr, UNKOWN};
    }
    return outer_map_blob2ptr_[blob_name];
  }

  /**
   * @brief Overrided from `IBlobsBuffer`, users could make tensorrt inference core use customed
   * data buffer to deploy inference. `data_ptr` and `location` are required to modify inner
   * mapping.
   *
   * @param blob_name The blob_name of model.
   * @param data_ptr Customed data buffer ptr.
   * @param location Where the data buffer locates.
   * @return true Successfully set customed data buffer.
   * @return false Will return false if `blob_name` does not match, or `data_ptr` is not valid.
   */
  bool SetBlobBuffer(const std::string &blob_name,
                     void              *data_ptr,
                     DataLocation       location) noexcept override
  {
    if (outer_map_blob2ptr_.find(blob_name) == outer_map_blob2ptr_.end())
    {
      LOG(ERROR) << "[TrtBlobBuffer] `SetBlobBuffer` Got invalid `blob_name`: " << blob_name;
      return false;
    }

    if (location == DataLocation::HOST)
    {
      outer_map_blob2ptr_[blob_name] = {inner_map_host_blob2ptr_[blob_name], location};
    } else
    {
      cudaPointerAttributes attr;
      cudaError_t           status = cudaPointerGetAttributes(&attr, data_ptr);
      if (status != cudaSuccess || attr.type != cudaMemoryType::cudaMemoryTypeDevice)
      {
        LOG(ERROR) << "[TrtBlobBuffer] `SetBlobBuffer` Got "
                      "invalid `data_ptr` "
                      "which should be "
                   << "allocated by `cudaMalloc`, but it "
                      "is NOT !!!";
        return false;
      }
      outer_map_blob2ptr_[blob_name] = {data_ptr, location};
    }
    return true;
  }

  /**
   * @brief Overrided from `IBlobsBuffer`, set the default buffer ptr used in tensorrt
   * engine inference stage. After calling `SetBlobBuffer`, `GetOuterBlobBuffer` could
   * get certain buffer ptr on `location`.
   *
   * @param blob_name The blob_name of model.
   * @param location Which buffer to use in inference stage.
   * @return true Successfully set blob buffer location.
   * @return false Will return false if blob_name does not match.
   */
  bool SetBlobBuffer(const std::string &blob_name, DataLocation location) noexcept override
  {
    if (outer_map_blob2ptr_.find(blob_name) == outer_map_blob2ptr_.end())
    {
      LOG(ERROR) << "[TrtBlobBuffer] `SetBlobBuffer` Got invalid `blob_name`: " << blob_name;
      return false;
    }

    outer_map_blob2ptr_[blob_name] = {
        (location == DataLocation::HOST ? inner_map_host_blob2ptr_[blob_name]
                                        : inner_map_device_blob2ptr_[blob_name]),
        location};

    return true;
  }

  /**
   * @brief Overrided from `IBlobsBuffer`, set the dynamic blob shape while tensorrt engine
   * doing inference. Note that `shape` should not has more element number than origin_shape
   * which is determined by model build stage. Dynamic shape suportted tensorrt inference
   * core should constructed by customed max blob shape params. There should not be `0` or any
   * negative values in `shape` vec.
   *
   * @note Please make sure your model supportes dynamic blob shape. Otherwise, it will leads
   * to unknown results.
   *
   * @param blob_name The blob_name of model.
   * @param shape Dynamic blob shape.
   * @return true
   * @return false Will return false if `shape` is not valid or `blob_name` does not match.
   */
  bool SetBlobShape(const std::string          &blob_name,
                    const std::vector<int64_t> &shape) noexcept override
  {
    if (map_blob_name2shape_.find(blob_name) == map_blob_name2shape_.end())
    {
      LOG(ERROR) << "[TrtBlobBuffer] `SetBlobShape` Got invalid `blob_name`: " << blob_name;
      return false;
    }
    const auto     &origin_shape      = map_blob_name2shape_[blob_name];
    const long long ori_element_count = CumVector(origin_shape);
    const long long dyn_element_count = CumVector(shape);
    if (origin_shape.size() != shape.size() || dyn_element_count > ori_element_count ||
        dyn_element_count < 0)
    {
      const std::string origin_shape_in_str = VisualVec(origin_shape);
      const std::string shape_in_str        = VisualVec(shape);
      LOG(ERROR) << "[TrtBlobBuffer] `SetBlobShape` Got invalid `shape` input. "
                 << "`shape`: " << shape_in_str << "\t"
                 << "`origin_shape`: " << origin_shape_in_str;
      return false;
    }
    map_blob_name2shape_[blob_name] = shape;
    return true;
  }

  /**
   * @brief Overrided from `IBlobsBuffer`, provide default or dynamic blob shape. Default
   * blob shape is defined while tensorrt inference core is built. Will return dynamic blob
   * shape if `SetBlobShape` is called before `GetBlobShape`.
   *
   * @param blob_name The blob_name of model.
   * @return const std::vector<int64_t>& . A const reference to blob shape recorded in buffer.
   */
  const std::vector<int64_t> &GetBlobShape(const std::string &blob_name) const noexcept override
  {
    if (map_blob_name2shape_.find(blob_name) == map_blob_name2shape_.end())
    {
      LOG(ERROR) << "[TrtBlobBuffer] `GetBlobShape` Got invalid `blob_name`: " << blob_name;
      static std::vector<int64_t> empty_shape;
      return empty_shape;
    }
    return map_blob_name2shape_.at(blob_name);
  }

  /**
   * @brief Overrided from `IBlobsBuffer`, provide number of blobs.
   *
   * @return size_t
   */
  size_t Size() const noexcept override
  {
    return outer_map_blob2ptr_.size();
  }

  /**
   * @brief Overrided from `IBlobsBuffer`, release the buffer instance.
   *
   */
  void Release() noexcept override
  {
    // release device buffer
    for (void *ptr : device_blobs_buffer_)
    {
      if (ptr != nullptr)
        cudaFree(ptr);
    }
    // release host buffer
    for (void *ptr : host_blobs_buffer_)
    {
      if (ptr != nullptr)
        delete[] reinterpret_cast<u_char *>(ptr);
    }
    device_blobs_buffer_.clear();
    host_blobs_buffer_.clear();
  }

  /**
   * @brief Overrided from `IBlobsBuffer`, reset the buffer instance which will not
   * release the buffer allocated. Mempool will call `Reset` after buffer instance is
   * returned by user.
   *
   */
  void Reset() noexcept override
  {
    for (const auto &p_name_ptr : inner_map_host_blob2ptr_)
    {
      outer_map_blob2ptr_[p_name_ptr.first] = {p_name_ptr.second, DataLocation::HOST};
    }
  }

  ~TrtBlobBuffer()
  {
    Release();
  }
  // no copy
  TrtBlobBuffer()                                 = default;
  TrtBlobBuffer(const TrtBlobBuffer &)            = delete;
  TrtBlobBuffer &operator=(const TrtBlobBuffer &) = delete;

  // mapping blob_name and buffer ptrs
  std::unordered_map<std::string, std::pair<void *, DataLocation>> outer_map_blob2ptr_;
  std::unordered_map<std::string, void *>                          inner_map_device_blob2ptr_;
  std::unordered_map<std::string, void *>                          inner_map_host_blob2ptr_;

  // buffer ptr vector, used while doing inference with tensorrt engine
  std::vector<void *> buffer_input_core_;

  // maintain buffer ptrs.
  std::vector<void *> device_blobs_buffer_;
  std::vector<void *> host_blobs_buffer_;

  // mapping blob_name and dynamic blob shape
  std::unordered_map<std::string, std::vector<int64_t>> map_blob_name2shape_;
};

} // namespace inference_core

#endif