/*
 * @Description:
 * @Author: Teddywesside 18852056629@163.com
 * @Date: 2024-11-25 18:38:34
 * @LastEditTime: 2024-12-02 19:03:30
 * @FilePath: /easy_deploy/deploy_core/include/deploy_core/base_sam.h
 */
#ifndef __EASY_DEPLOY_BASE_SAM_H
#define __EASY_DEPLOY_BASE_SAM_H

#include "deploy_core/base_infer_core.h"
#include "deploy_core/common_defination.h"

#include <opencv2/opencv.hpp>

namespace sam {

/**
 * @brief The common sam pipeline package wrapper.
 *
 */
struct SamPipelinePackage : public async_pipeline::IPipelinePackage {
  // maintain image-encoder's blobs buffer
  std::shared_ptr<inference_core::IBlobsBuffer> image_encoder_blobs_buffer;
  // maintain mask-decoder's blobs buffer
  std::shared_ptr<inference_core::IBlobsBuffer> mask_decoder_blobs_buffer;

  // the wrapped pipeline image data
  std::shared_ptr<async_pipeline::IPipelineImageData> input_image_data;
  // input boxes prompt
  std::vector<BBox2D> boxes;
  // input points prompt
  std::vector<std::pair<int, int>> points;
  // input points labels
  std::vector<int> labels;
  // record the transform factor in image preprocessing
  float transform_scale;
  // mask results
  cv::Mat mask;

  // the blobs buffer used in inference core processing
  std::shared_ptr<inference_core::IBlobsBuffer> infer_buffer;
  std::shared_ptr<inference_core::IBlobsBuffer> GetInferBuffer() override
  {
    return infer_buffer;
  }
};

/**
 * @brief The abstract interface class of `Segment Anything Model`(SAM) which defines
 * image-preprocess、prompt-preprocess、mask-postprocess interfaces. Any SAM algorithms
 * implementation could override these pure virtual methods to make up a sync/async
 * inference supported pipeline.
 *
 * workflow:
 *
 * `ImagePreProcess` --> `ImageEncoderInfer` --> `PromptBoxPreProcess`/`PromptPointPreProcess`
 * --> `MaskDecoderInfer` --> `MaskPostProcess`
 *
 */
class ISamModel {
protected:
  typedef std::shared_ptr<async_pipeline::IPipelinePackage> ParsingType;
  virtual ~ISamModel() = default;
  /**
   * @brief The `ImagePreProcess` stage. Inside the method, you should cast the `pipeline_unit`
   * pointer to `SamPipelinePackage` type pointer, and check if the convertion works. If the
   * package pointer is not valid or anything goes wrong, it should return `false` to mention
   * the inference pipelinee to drop the package.
   *
   * @param pipeline_unit
   * @return true
   * @return false
   */
  virtual bool ImagePreProcess(ParsingType pipeline_unit) = 0;

  /**
   * @brief The `PromptBoxPreProcess` stage. Inside the method, you should cast the `pipeline_unit`
   * pointer to `SamPipelinePackage` type pointer, and check if the convertion works. If the
   * package pointer is not valid or anything goes wrong, it should return `false` to mention
   * the inference pipelinee to drop the package.
   *
   * @param pipeline_unit
   * @return true
   * @return false
   */
  virtual bool PromptBoxPreProcess(ParsingType pipeline_unit) = 0;

  /**
   * @brief The `PromptPointPreProcess` stage. Inside the method, you should cast the
   * `pipeline_unit` pointer to `SamPipelinePackage` type pointer, and check if the convertion
   * works. If the package pointer is not valid or anything goes wrong, it should return `false` to
   * mention the inference pipelinee to drop the package.
   *
   * @param pipeline_unit
   * @return true
   * @return false
   */
  virtual bool PromptPointPreProcess(ParsingType pipeline_unit) = 0;

  /**
   * @brief The `MaskPostProcess` stage. Inside the method, you should cast the `pipeline_unit`
   * pointer to `SamPipelinePackage` type pointer, and check if the convertion works. If the
   * package pointer is not valid or anything goes wrong, it should return `false` to mention
   * the inference pipelinee to drop the package.
   *
   * @param pipeline_unit
   * @return true
   * @return false
   */
  virtual bool MaskPostProcess(ParsingType pipeline_unit) = 0;
};

/**
 * @brief A functor to generate sam results from `SamPipelinePackage`. Used in async pipeline.
 *
 */
class SamGenResultType {
public:
  cv::Mat operator()(const std::shared_ptr<async_pipeline::IPipelinePackage> &package)
  {
    auto sam_package = std::dynamic_pointer_cast<SamPipelinePackage>(package);
    if (sam_package == nullptr)
    {
      LOG(ERROR) << "[SamGenResultType] Got INVALID package ptr!!!";
      return {};
    }
    return std::move(sam_package->mask);
  }
};

/**
 * @brief The base class of SAM model. It implements `GenerateMask` and `GenerateMaskAsync`
 * both with `box` prompts or `points` prompts. In the asynchronous pipeline inference mode,
 * the `box` pipeline and `point` pipeline could been used in the same time, cause they are
 * independent.
 *
 */
class BaseSamModel : public ISamModel,
                     public async_pipeline::BaseAsyncPipeline<cv::Mat, SamGenResultType> {
protected:
  using ParsingType = std::shared_ptr<async_pipeline::IPipelinePackage>;
  /**
   * @brief Construct `BaseSamModel` with `image_encoder_core` and at least one of `mask_points_
   * decoder_core` or `mask_boxes_decoder_core`. Will throw exception if both decoders with points
   * and boxes are nullptr.
   *
   * @param model_name
   * @param image_encoder_core
   * @param mask_points_decoder_core
   * @param mask_boxes_decoder_core
   */
  BaseSamModel(const std::string                             &model_name,
               std::shared_ptr<inference_core::BaseInferCore> image_encoder_core,
               std::shared_ptr<inference_core::BaseInferCore> mask_points_decoder_core,
               std::shared_ptr<inference_core::BaseInferCore> mask_boxes_decoder_core);

  virtual ~BaseSamModel();

public:
  /**
   * @brief Generate the mask with points as prompts in sync mode.
   *
   * @param image input image
   * @param points points coords
   * @param labels points labels, 0 - background; 1 - foreground
   * @param cv::Mat reference to the result. 0 - background; 255 - foreground
   * @param isRGB if the input image is RGB format. default=false
   * @return true
   * @return false
   */
  bool GenerateMask(const cv::Mat                          &image,
                    const std::vector<std::pair<int, int>> &points,
                    const std::vector<int>                 &labels,
                    cv::Mat                                &result,
                    bool                                    isRGB = false);
  /**
   * @brief Generate the mask with boxes as prompts in sync mode.
   *
   * @note SAM model with boxes only support one box as its prompts. More boxes wont make any
   * exception, but also will not take effect.
   *
   * @param image input image
   * @param boxes boxes coords
   * @param cv::Mat reference to the result. 0 - background; 255 - foreground
   * @param isRGB if the input image is RGB format. default=false
   * @return true
   * @return false
   */
  bool GenerateMask(const cv::Mat             &image,
                    const std::vector<BBox2D> &boxes,
                    cv::Mat                   &result,
                    bool                       isRGB = false);

  /**
   * @brief Generate the mask with points as prompts in async mode.
   *
   * @warning The returned `std::future<>` instance could be invalid. Please make sure it is
   * valid before you call `get()`.
   *
   * @param image input image
   * @param points points coords
   * @param labels points labels, 0 - background; 1 - foreground
   * @param isRGB if the input image is RGB format. default=false
   * @param cover_oldest whether cover the oldest package if the pipeline queue is full.
   * default=false.
   * @return std::future<cv::Mat> A std::future instance of the result.
   */
  [[nodiscard]] std::future<cv::Mat> GenerateMaskAsync(
      const cv::Mat                          &image,
      const std::vector<std::pair<int, int>> &points,
      const std::vector<int>                 &labels,
      bool                                    isRGB        = false,
      bool                                    cover_oldest = false);

  /**
   * @brief Generate the mask with boxes as prompts in async mode.
   *
   * @note SAM model with boxes only support one box as its prompts. More boxes wont make any
   * exception, but also will not take effect.
   *
   * @warning The returned `std::future<>` instance could be invalid. Please make sure it is
   * valid before you call `get()`.
   *
   * @param image input image
   * @param boxes boxes coords
   * @param callback callback function if needed. default=nullptr.
   * @param isRGB if the input image is RGB format. default=false
   * @param cover_oldest whether cover the oldest package if the pipeline queue is full.
   * default=false.
   * @return std::future<cv::Mat> A std::future instance of the result.
   */
  [[nodiscard]] std::future<cv::Mat> GenerateMaskAsync(const cv::Mat             &image,
                                                       const std::vector<BBox2D> &boxes,
                                                       bool                       isRGB = false,
                                                       bool cover_oldest                = false);

private:
  // forbidden the access from outside to `BaseAsyncPipeline::PushPipeline`
  using BaseAsyncPipeline::PushPipeline;

  void ConfigureBoxPipeline();

  void ConfigurePointPipeline();

protected:
  std::shared_ptr<inference_core::BaseInferCore> image_encoder_core_;
  std::shared_ptr<inference_core::BaseInferCore> mask_points_decoder_core_;
  std::shared_ptr<inference_core::BaseInferCore> mask_boxes_decoder_core_;

  const std::string box_pipeline_name_;
  const std::string point_pipeline_name_;
  const std::string model_name_;
};


/**
 * @brief Abstract factory base class of Sam model.
 * 
 */
class BaseSamFactory {
public:
  virtual std::shared_ptr<sam::BaseSamModel> Create() = 0;
};

} // namespace sam

#endif