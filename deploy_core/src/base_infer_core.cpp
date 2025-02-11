/*
 * @Description:
 * @Author: Teddywesside 18852056629@163.com
 * @Date: 2024-11-19 18:33:00
 * @LastEditTime: 2024-11-26 21:56:31
 * @FilePath: /easy_deploy/deploy_core/src/base_infer_core.cpp
 */
#include "deploy_core/base_infer_core.h"

namespace inference_core {

// used in sync infer
struct _InnerSyncInferPackage : public async_pipeline::IPipelinePackage {
public:
  std::shared_ptr<IBlobsBuffer> GetInferBuffer() override
  {
    return buffer;
  }
  std::shared_ptr<IBlobsBuffer> buffer;
};

BaseInferCore::BaseInferCore()
{
  auto preprocess_block = BuildPipelineBlock(
      [&](ParsingType unit) -> bool { return PreProcess(unit); }, "BaseInferCore PreProcess");
  auto inference_block = BuildPipelineBlock(
      [&](ParsingType unit) -> bool { return Inference(unit); }, "BaseInferCore Inference");
  auto postprocess_block = BuildPipelineBlock(
      [&](ParsingType unit) -> bool { return PostProcess(unit); }, "BaseInferCore PostProcess");
  ConfigPipeline("InferCore Pipieline", {preprocess_block, inference_block, postprocess_block});
}

bool BaseInferCore::SyncInfer(std::shared_ptr<IBlobsBuffer> buffer, const int batch_size)
{
  auto inner_package    = std::make_shared<_InnerSyncInferPackage>();
  inner_package->buffer = buffer;
  CHECK_STATE(PreProcess(inner_package), "[BaseInferCore] SyncInfer Preprocess Failed!!!");
  CHECK_STATE(Inference(inner_package), "[BaseInferCore] SyncInfer Inference Failed!!!");
  CHECK_STATE(PostProcess(inner_package), "[BaseInferCore] SyncInfer PostProcess Failed!!!");
  return true;
}

std::shared_ptr<IBlobsBuffer> BaseInferCore::GetBuffer(bool block)
{
  return mem_buf_pool_->Alloc(block);
}

void BaseInferCore::Release()
{
  BaseAsyncPipeline::ClosePipeline();
  mem_buf_pool_.reset();
}

void BaseInferCore::Init(int mem_buf_size)
{
  if (mem_buf_size <= 0 || mem_buf_size > 100)
  {
    throw std::invalid_argument("mem_buf_size should be between [1,100], Got: " +
                                std::to_string(mem_buf_size));
  }
  mem_buf_pool_ = std::make_unique<MemBufferPool>(this, mem_buf_size);
  LOG(INFO) << "successfully init mem buf pool with pool_size : " << mem_buf_size;
}

BaseInferCore::~BaseInferCore()
{
  Release();
}

} // namespace inference_core