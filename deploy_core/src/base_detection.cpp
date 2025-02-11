/*
 * @Description: 
 * @Author: Teddywesside 18852056629@163.com
 * @Date: 2024-11-25 14:24:19
 * @LastEditTime: 2024-11-26 21:58:50
 * @FilePath: /easy_deploy/deploy_core/src/base_detection.cpp
 */
#include "deploy_core/base_detection.h"

#include "deploy_core/wrapper.h"

namespace detection_2d {

std::string BaseDetectionModel::detection_pipeline_name_ = "DetectionPipeline";

/**
 * @brief construct a `DetectionPipelinePackage`
 *
 * @param input_image
 * @param conf_thresh
 * @param isRGB
 * @param blob_buffers
 * @return std::shared_ptr<DetectionPipelinePackage>
 */
static std::shared_ptr<DetectionPipelinePackage> CreateDetectionPipelineUnit(
    const cv::Mat                &input_image,
    float                         conf_thresh,
    bool                          isRGB,
    std::shared_ptr<inference_core::IBlobsBuffer> blob_buffers)
{
  // 1. construct the image wrapper
  auto image_wrapper = std::make_shared<PipelineCvImageWrapper>(input_image, isRGB);
  // 2. construct `DetectionPipelinePakcage`
  auto package              = std::make_shared<DetectionPipelinePackage>();
  package->input_image_data = image_wrapper;
  package->conf_thresh      = conf_thresh;
  package->infer_buffer     = blob_buffers;

  return package;
}

BaseDetectionModel::BaseDetectionModel(std::shared_ptr<inference_core::BaseInferCore> infer_core)
    : infer_core_(infer_core)
{
  // 1. check infer_core
  if (infer_core == nullptr)
  {
    throw std::invalid_argument("[BaseDetectionModel] Input argument `infer_core` is nullptr!!!");
  }

  // 2. configure pipeline
  auto preprocess_block = BaseAsyncPipeline::BuildPipelineBlock(
      [=](ParsingType unit) -> bool { return PreProcess(unit); }, "BaseDet PreProcess");

  auto infer_core_context = infer_core->GetPipelineContext();

  auto postprocess_block = BaseAsyncPipeline::BuildPipelineBlock(
      [=](ParsingType unit) -> bool { return PostProcess(unit); }, "BaseDet PostProcess");

  BaseAsyncPipeline::ConfigPipeline(detection_pipeline_name_,
                                    {preprocess_block, infer_core_context, postprocess_block});
}

bool BaseDetectionModel::Detect(const cv::Mat       &input_image,
                                std::vector<BBox2D> &det_results, // todo
                                float                conf_thresh,
                                bool                 isRGB) noexcept
{
  // 1. Get blobs buffer
  auto blob_buffers = infer_core_->GetBuffer(false);
  if (blob_buffers == nullptr)
  {
    LOG(ERROR) << "[BaseDetectionModel] Inference Core run out buffer!!!";
    return false;
  }

  // 2. Create a dummy pipeline package
  auto package = CreateDetectionPipelineUnit(input_image, conf_thresh, isRGB, blob_buffers);

  // 3. preprocess by derived class
  MESSURE_DURATION_AND_CHECK_STATE(PreProcess(package),
                                   "[BaseDetectionModel] Preprocess execute failed!!!");

  // 4. network inference
  MESSURE_DURATION_AND_CHECK_STATE(infer_core_->SyncInfer(blob_buffers),
                                   "[BaseDetectionModel] SyncInfer execute failed!!!");

  // 5. postprocess by derived class
  MESSURE_DURATION_AND_CHECK_STATE(PostProcess(package),
                                   "[BaseDetectionModel] PostProcess execute failed!!!");

  // 6. take output
  det_results = std::move(package->results);

  return true;
}

std::future<std::vector<BBox2D>> BaseDetectionModel::DetectAsync(const cv::Mat &input_image,
                                                                 float          conf_thresh,
                                                                 bool           isRGB,
                                                                 bool cover_oldest) noexcept
{
  // 1. check if the pipeline is initialized
  if (!IsPipelineInitialized(detection_pipeline_name_))
  {
    LOG(ERROR) << "[BaseDetectionModel] Async Pipeline is not init yet!!!";
    return std::future<std::vector<BBox2D>>();
  }

  // 2. get blob buffer
  auto blob_buffers = infer_core_->GetBuffer(true);
  if (blob_buffers == nullptr)
  {
    LOG(ERROR) << "[BaseDetectionModel] Failed to get buffer from inference core!!!";
    return std::future<std::vector<BBox2D>>();
  }

  // 3. create a pipeline package
  auto package = CreateDetectionPipelineUnit(input_image, conf_thresh, isRGB, blob_buffers);

  // 4. push package into pipeline and return `std::future`
  return PushPipeline(detection_pipeline_name_, package);
}

BaseDetectionModel::~BaseDetectionModel()
{
  ClosePipeline();
  infer_core_->Release();
}


} // namespace detection_2d