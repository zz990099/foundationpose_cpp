#pragma once

#include <tuple>
#include <opencv2/opencv.hpp>
#include <glog/logging.h>
#include <glog/log_severity.h>
#include <Eigen/Dense>
#include <fstream>

inline std::tuple<cv::Mat, cv::Mat, cv::Mat> ReadRgbDepthMask(const std::string& rgb_path,
                                                       const std::string& depth_path,
                                                       const std::string& mask_path)
{
  cv::Mat rgb = cv::imread(rgb_path);
  cv::Mat depth = cv::imread(depth_path, cv::IMREAD_UNCHANGED);
  cv::Mat mask = cv::imread(mask_path, cv::IMREAD_UNCHANGED);

  CHECK(!rgb.empty()) << "Failed reading rgb from path : " << rgb_path;
  CHECK(!depth.empty()) << "Failed reading depth from path : " << depth_path;
  CHECK(!mask.empty()) << "Failed reading mask from path : " << mask_path;

  depth.convertTo(depth, CV_32FC1);
  depth = depth / 1000.f;

  cv::cvtColor(rgb, rgb, cv::COLOR_BGR2RGB);
  if (mask.channels() == 3) {
    cv::cvtColor(mask, mask, cv::COLOR_BGR2RGB);
    std::vector<cv::Mat> channels;
    cv::split(mask, channels);
    mask = channels[0];
  }

  return {rgb, depth, mask};
}

inline std::tuple<cv::Mat, cv::Mat> ReadRgbDepth(const std::string& rgb_path,
                                          const std::string& depth_path)
{
  cv::Mat rgb = cv::imread(rgb_path);
  cv::Mat depth = cv::imread(depth_path, cv::IMREAD_UNCHANGED);

  CHECK(!rgb.empty()) << "Failed reading rgb from path : " << rgb_path;
  CHECK(!depth.empty()) << "Failed reading depth from path : " << depth_path;

  depth.convertTo(depth, CV_32FC1);
  depth = depth / 1000.f;

  cv::cvtColor(rgb, rgb, cv::COLOR_BGR2RGB);

  return {rgb, depth};
}

inline void draw3DBoundingBox(const Eigen::Matrix3f& intrinsic, 
                       const Eigen::Matrix4f& pose, 
                       int input_image_H, 
                       int input_image_W, 
                       const Eigen::Vector3f& dimension,
                       cv::Mat& image) {
    // 目标的长宽高
    float l = dimension(0) / 2;
    float w = dimension(1) / 2;
    float h = dimension(2) / 2;

    // 目标的八个顶点在物体坐标系中的位置
    Eigen::Vector3f points[8] = {
        {-l, -w, h}, {l, -w, h}, {l, w, h}, {-l, w, h},
        {-l, -w, -h}, {l, -w, -h}, {l, w, -h}, {-l, w, -h}
    };


    // 变换到世界坐标系
    Eigen::Vector4f transformed_points[8];
    for (int i = 0; i < 8; ++i) {
        transformed_points[i] = pose * Eigen::Vector4f(points[i](0), points[i](1), points[i](2), 1);
    }

    // 投影到图像平面
    std::vector<cv::Point2f> image_points;
    for (int i = 0; i < 8; ++i) {
        float x = transformed_points[i](0) / transformed_points[i](2);
        float y = transformed_points[i](1) / transformed_points[i](2);

        // 使用内参矩阵进行投影
        float u = intrinsic(0, 0) * x + intrinsic(0, 2);
        float v = intrinsic(1, 1) * y + intrinsic(1, 2);

        image_points.emplace_back(static_cast<float>(u), static_cast<float>(v));

    }

    // 绘制边框（连接顶点）
    std::vector<std::pair<int, int>> edges = {
        {0, 1}, {1, 2}, {2, 3}, {3, 0}, // 底面
        {4, 5}, {5, 6}, {6, 7}, {7, 4}, // 顶面
        {0, 4}, {1, 5}, {2, 6}, {3, 7}  // 侧面
    };

    for (const auto& edge : edges) {
        if (edge.first < image_points.size() && edge.second < image_points.size()) {
            cv::line(image, image_points[edge.first], image_points[edge.second], 
                     cv::Scalar(0, 255, 0), 2); // 绿色边框
        }
    }
}

inline Eigen::Matrix3f ReadCamK(const std::string& cam_K_path)
{
  Eigen::Matrix3f K;

  // 打开文件
  std::ifstream file(cam_K_path.c_str());
  CHECK(file) << "Failed open file : " << cam_K_path;

  // 读取数据并存入矩阵
  for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
          file >> K(i, j);
      }
  }

  // 关闭文件
  file.close();

  return K;
}

inline void saveVideo(const std::vector<cv::Mat>& frames, const std::string& outputPath, double fps = 30.0) {
    if (frames.empty()) {
        std::cerr << "Error: No frames to write!" << std::endl;
        return;
    }

    // 获取帧的宽度和高度
    int frameWidth = frames[0].cols;
    int frameHeight = frames[0].rows;
    
    // 定义视频编码格式（MP4 使用 `cv::VideoWriter::fourcc('m', 'p', '4', 'v')` 或 `cv::VideoWriter::fourcc('H', '2', '6', '4')`）
    int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');  // MPEG-4 编码
    // int fourcc = cv::VideoWriter::fourcc('H', '2', '6', '4'); // H.264 编码（可能需要额外的编解码器支持）

    // 创建 VideoWriter 对象
    cv::VideoWriter writer(outputPath, fourcc, fps, cv::Size(frameWidth, frameHeight));

    // 检查是否成功打开
    if (!writer.isOpened()) {
        std::cerr << "Error: Could not open the video file for writing!" << std::endl;
        return;
    }

    // 写入所有帧
    for (const auto& frame : frames) {
        // 确保所有帧大小一致
        if (frame.cols != frameWidth || frame.rows != frameHeight) {
            std::cerr << "Error: Frame size mismatch!" << std::endl;
            break;
        }
        writer.write(frame);
    }

    // 释放资源
    writer.release();
    std::cout << "Video saved successfully: " << outputPath << std::endl;
}