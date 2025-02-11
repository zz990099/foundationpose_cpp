/*
 * @Description:
 * @Author: Teddywesside 18852056629@163.com
 * @Date: 2024-11-25 14:00:38
 * @LastEditTime: 2024-11-26 21:58:32
 * @FilePath: /easy_deploy/deploy_core/include/deploy_core/wrapper.h
 */
#ifndef __EASY_DEPLOY_WRAPPER_H
#define __EASY_DEPLOY_WRAPPER_H

#include "deploy_core/async_pipeline.h"

#include <opencv2/opencv.hpp>

#include <unordered_map>

/**
 * @brief A simple wrapper of cv::Mat. Used in pipeline.
 *
 */
class PipelineCvImageWrapper : public async_pipeline::IPipelineImageData {
public:
  PipelineCvImageWrapper(const cv::Mat &cv_image, bool isRGB = false) : inner_cv_image(cv_image)
  {
    image_data_info.data_pointer   = cv_image.data;
    image_data_info.format         = isRGB ? ImageDataFormat::RGB : ImageDataFormat::BGR;
    image_data_info.image_height   = cv_image.rows;
    image_data_info.image_width    = cv_image.cols;
    image_data_info.image_channels = cv_image.channels();
    image_data_info.location       = DataLocation::HOST;
  }

  const ImageDataInfo &GetImageDataInfo() const
  {
    return image_data_info;
  }

private:
  IPipelineImageData::ImageDataInfo image_data_info;
  const cv::Mat                     inner_cv_image;
};

#endif