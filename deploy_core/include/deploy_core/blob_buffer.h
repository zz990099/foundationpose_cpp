/*
 * @Description:
 * @Author: Teddywesside 18852056629@163.com
 * @Date: 2024-11-25 15:27:59
 * @LastEditTime: 2024-11-26 21:57:59
 * @FilePath: /easy_deploy/deploy_core/include/deploy_core/blob_buffer.h
 */
#ifndef __EASY_DEPLOY_BLOB_BUFFER_H
#define __EASY_DEPLOY_BLOB_BUFFER_H

#include <memory>
#include <vector>

#include "deploy_core/common_defination.h"

namespace inference_core {

/**
 * @brief The key to abstracting and shielding the inference framework and hardware characteristics
 * lies in how the management of inference buffer is abstracted. Considering the requirements
 * of asynchronous inference framework, we encapsulated the buffer used during inference into a
 * dedicated class and abstracted its functionality by developing the `IBlobsBuffer` interface
 * class. The implementation of this interface must provide the following functionalities:
 *
 *  1. Set the buffer address to be used for inference.
 *
 *  2. Get the default buffer address.
 *
 *  3. Set the shape of the model blob.
 *
 *  4. Get the default blob shape.
 *
 * And Other base functionalities as declared below.
 *
 */
class IBlobsBuffer {
public:
  /**
   * @brief The `BlobsBuffer` instance should provide the buffer ptr which will be used in the
   * inference process. This buffer is allocated by certain inference_core by default. User could
   * customize the buffer ptr by calling `SetBlobBuffer`.
   *
   * @param blob_name The name of the blob.
   * @return std::pair<void*, DataLocation> Will return {nullptr, UNKOWN} if `blob_name` is
   * invalid.
   */
  virtual std::pair<void *, DataLocation> GetOuterBlobBuffer(
      const std::string &blob_name) noexcept = 0;

  /**
   * @brief The `BlobsBuffer` instance should provide the functionality to accept a customized
   * data buffer ptr which could be on host or device. Some inference frameworks based on
   * heterogeneous architecture hardware (e.g. CUDA) use buffer on device to deploy inference. There
   * is no need to copy data from host to device if the device buffer ptr is provided to
   * `BlobsBuffer`.
   *
   * @param blob_name The name of the blob.
   * @param data_ptr The ptr of the customized data buffer.
   * @param location Location of the customized data buffer.
   * @return true
   * @return false Will return false if `blob_name` is invalid.
   */
  virtual bool SetBlobBuffer(const std::string &blob_name,
                             void              *data_ptr,
                             DataLocation       location) noexcept = 0;

  /**
   * @brief `SetBlobBuffer` provides the functionality to change the default using data buffer
   * on host size or device side. After calling this method, `GetOuterBlobBuffer` will return
   * the buffer ptr on the certain side.
   *
   * @note Some inference frameworks (e.g. onnxruntime, rknn) do not distinguish buffer between
   * the host side and the device side. So this method will not change their default buffer ptr.
   *
   * @param blob_name The name of the blob.
   * @param location Location of the customized data buffer.
   * @return true
   * @return false Will return false if `blob_name` is invalid.
   */
  virtual bool SetBlobBuffer(const std::string &blob_name, DataLocation location) noexcept = 0;

  /**
   * @brief `SetBlobShape` provides the functionality to change the dynamic blob shape in the
   * inference processing if the model engine allows.
   *
   * @note Some inference framework (e.g. rknn) do not support dynamic blob shape. And make sure
   * your model supports dynamic blob shape before you call this method.
   *
   * @param blob_name The name of the blob.
   * @param shape The dynamic blob shape.
   * @return true
   * @return false Will return false if `blob_name` is invalid.
   */
  virtual bool SetBlobShape(const std::string          &blob_name,
                            const std::vector<int64_t> &shape) noexcept = 0;

  /**
   * @brief `GetBlobShape` provides the functionality to get the dynamic blob shape in the
   * inference processing. By default, this will return the max blob shape which is parsed
   * in `inference_core` construction.
   *
   * @param blob_name The name of the blob.
   * @return const std::vector<int64_t>& The const reference of blob shape vector maintained.
   */
  virtual const std::vector<int64_t> &GetBlobShape(const std::string &blob_name) const noexcept = 0;

  /**
   * @brief Return the total number of blobs.
   *
   * @return size_t
   */
  virtual size_t Size() const noexcept = 0;

  /**
   * @brief Reset the `BlobsBuffer` which will not release the buffer memory.
   *
   */
  virtual void Reset() noexcept = 0;

protected:
  virtual ~IBlobsBuffer() noexcept = default;

  /**
   * @brief Release the whole `BlobsBuffer` instance.
   *
   */
  virtual void Release() noexcept = 0;
};

} // namespace inference_core

#endif