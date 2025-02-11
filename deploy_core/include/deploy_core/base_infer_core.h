/*
 * @Description:
 * @Author: Teddywesside 18852056629@163.com
 * @Date: 2024-11-26 08:42:05
 * @LastEditTime: 2024-12-02 19:03:37
 * @FilePath: /easy_deploy/deploy_core/include/deploy_core/base_infer_core.h
 */
#ifndef __EASY_DEPLOY_BASE_INFER_CORE_H
#define __EASY_DEPLOY_BASE_INFER_CORE_H

#include <atomic>
#include <memory>
#include <thread>
#include <vector>

#include "deploy_core/block_queue.h"
#include "deploy_core/async_pipeline.h"

namespace inference_core {

enum InferCoreType { ONNXRUNTIME, TENSORRT, RKNN, NOT_PROVIDED };

/**
 * @brief `IRotInferCore` is abstract interface class which defines all pure virtual functions
 * that the derived class should implement, e.g., `PreProcess`, `Inference` and `PostProcess`.
 *
 */
class IRotInferCore {
public:
  /**
   * @brief `AllocBlobsBuffer` is a common interface that user could get a brand new buffer
   * instance by. This pure virtual function is implemented by actual inference core, which
   * may take a while to process. Use pre-allocated buffer instance in mem buffer pool could
   * get better performance. See `BaseInferCore`.
   *
   * @return std::shared_ptr<IBlobsBuffer> A brand new buffer instance allocated by inference
   * core.
   */
  virtual std::shared_ptr<IBlobsBuffer> AllocBlobsBuffer() = 0;

  /**
   * @brief Get the core type.
   *
   * @return InferCoreType
   */
  virtual InferCoreType GetType()
  {
    return InferCoreType::NOT_PROVIDED;
  }

  /**
   * @brief Return the name of inference core.
   *
   * @return std::string
   */
  virtual std::string GetName()
  {
    return "";
  }

protected:
  virtual ~IRotInferCore() = default;

  /**
   * @brief `PreProcess` stage of the inference process. Return true if this is stage is not
   * needed in the actual inference core implementation. Return false if something went wrong
   * while doing processing. The pipeline will drop the package if `PreProcess` returns false.
   *
   * @param buffer a common "pipeline" package ptr.
   * @return true
   * @return false
   */
  virtual bool PreProcess(std::shared_ptr<async_pipeline::IPipelinePackage> buffer) = 0;

  /**
   * @brief `Inference` stage of the inference process. Return false if something went wrong
   * while doing processing. The pipeline will drop the package if `Inference` returns false.
   *
   * @param buffer a common "pipeline" package ptr.
   * @return true
   * @return false
   */
  virtual bool Inference(std::shared_ptr<async_pipeline::IPipelinePackage> buffer) = 0;

  /**
   * @brief `PostProcess` stage of the inference process. Return false if something went wrong
   * while doing processing. The pipeline will drop the package if `PostProcess` returns false.
   *
   * @param buffer a common "pipeline" package ptr.
   * @return true
   * @return false
   */
  virtual bool PostProcess(std::shared_ptr<async_pipeline::IPipelinePackage> buffer) = 0;
};

/**
 * @brief A simple implementation of mem buffer pool. Using `BlockQueue` to deploy a producer-
 * consumer model. It will allocate buffer using `AllocBlobsBuffer` method of `IRotInferCore`
 * and provides `IBlobsBuffer` ptr when `Alloc` method is called. The "Alloced" buffer will
 * return back to mem buffer pool while the customed deconstruction method of shared_ptr ptr
 * is called.
 *
 */
class MemBufferPool {
public:
  MemBufferPool(IRotInferCore *infer_core, const int pool_size)
      : pool_size_(pool_size), dynamic_pool_(pool_size)
  {
    for (int i = 0; i < pool_size; ++i)
    {
      auto blob_buffer = infer_core->AllocBlobsBuffer();
      dynamic_pool_.BlockPush(blob_buffer.get());
      static_pool_.insert({blob_buffer.get(), blob_buffer});
    }
  }

  std::shared_ptr<IBlobsBuffer> Alloc(bool block)
  {
    // customed deconstruction method
    auto func_dealloc = [&](IBlobsBuffer *buf) {
      buf->Reset();
      this->dynamic_pool_.BlockPush(buf);
    };

    auto buf = block ? dynamic_pool_.Take() : dynamic_pool_.TryTake();
    return buf.has_value() ? std::shared_ptr<IBlobsBuffer>(buf.value(), func_dealloc) : nullptr;
  }

  void Release()
  {
    if (dynamic_pool_.Size() != pool_size_)
    {
      LOG(WARNING) << "[MemBufPool] does not maintain all bufs when release func called!";
    }
    static_pool_.clear();
  }

  int RemainSize()
  {
    return dynamic_pool_.Size();
  }

  ~MemBufferPool()
  {
    Release();
  }

private:
  const int                                                         pool_size_;
  BlockQueue<IBlobsBuffer *>                                        dynamic_pool_;
  std::unordered_map<IBlobsBuffer *, std::shared_ptr<IBlobsBuffer>> static_pool_;
};

/**
 * @brief A dummy class to help `BaseInferCore` inherit from `BaseAsyncPipeline` to generate
 * async pipeline framework.
 *
 */
class _DummyInferCoreGenReulstType {
public:
  bool operator()(const std::shared_ptr<async_pipeline::IPipelinePackage> & /*package*/)
  {
    return true;
  }
};

/**
 * @brief `BaseInferCore` inherits `IRotInferCore` and `BaseAsyncPipeline`. `IRotInferCore`
 * defines all pure virtual methods of the abstract function of the inference core.
 * `BaseAsyncPipeline` provides a set of methods to help user build and utilize a async
 * inference pipeline. See `BaseAsyncPipeline` defination.
 *
 * @note The inheritance relationship between class A and class B is modified by protected.
 * And `BaseInferCore` only makes the `GetPipelineContext` method public, which means the
 * derived class of `BaseInferCore` is not supported to deploy async pipeline inference
 * process. It should be used by specific algorithms in its entirety.
 *
 */
class BaseInferCore : public IRotInferCore,
                      protected async_pipeline::BaseAsyncPipeline<bool, _DummyInferCoreGenReulstType> {
protected:
  BaseInferCore();
  typedef std::shared_ptr<async_pipeline::IPipelinePackage> ParsingType;

public:
  using BaseAsyncPipeline::GetPipelineContext;

  /**
   * @brief This function provides a sync inference process which is completely independent
   * of the async inference pipeline. Through, it depends on the three stage virtual methods
   * defined in `IRotInferCore`. Return false if something went wrong while inference.
   *
   * @param buffer
   * @param batch_size default=1, multi-batch inference may not be supported.
   * @return true
   * @return false
   */
  bool SyncInfer(std::shared_ptr<IBlobsBuffer> buffer, const int batch_size = 1);

  /**
   * @brief Get the pre-allocated blobs buffer shared pointer. The returned pointer is a
   * smart pointer which will automatically return to the pool when it is released.
   *
   * @param block whether to block the thread if the pool is empty.
   * @return std::shared_ptr<IBlobsBuffer>
   */
  std::shared_ptr<IBlobsBuffer> GetBuffer(bool block);

  /**
   * @brief Release the sources in base class.
   *
   * @warning The derived class should call `BaseInferCore::Release()` in its deconstruct
   * function in order to release the blobs buffer before the enviroment is destroyed.
   * Things go wrong if allocated memory released after their enviroment released on some
   * hardware.
   *
   */
  virtual void Release();

protected:
  virtual ~BaseInferCore();

  /**
   * @brief Init the base class memory pool.
   *
   * @warning Please call `Init()` at the derived class construct function`s end when the
   * runtime enviroment is setup successfully. This method will call `AllocBlobsBuffer`
   * to create a memory pool. Temporary we manually call this method to init the memory pool.
   *
   * @param mem_buf_size number of blobs buffers pre-allocated.
   */
  void Init(int mem_buf_size = 5);

private:
  std::unique_ptr<MemBufferPool> mem_buf_pool_{nullptr};
};

/**
 * @brief Abstract factory class of infer_core.
 * 
 */
class BaseInferCoreFactory {
public:
  virtual std::shared_ptr<inference_core::BaseInferCore> Create() = 0;
};

} // namespace inference_core

#endif