#include "tests/fps_counter.h"
#include <glog/logging.h>
#include <glog/log_severity.h>
#include <gtest/gtest.h>
#include <opencv4/opencv2/opencv.hpp>
#include <fstream>

#include "detection_6d_foundationpose/foundationpose.hpp"
#include "trt_core/trt_core.h"

using namespace inference_core;
using namespace detection_6d;

static const std::string refiner_engine_path_ = "/workspace/models/refiner_hwc_dynamic_fp16.engine";
static const std::string scorer_engine_path_ = "/workspace/models/scorer_hwc_dynamic_fp16.engine";
static const std::string demo_data_path_ = "/workspace/test_data/mustard0";
static const std::string demo_textured_obj_path = demo_data_path_ + "/mesh/textured_simple.obj";
static const std::string demo_textured_map_path = demo_data_path_ + "/mesh/texture_map.png";
static const std::string demo_name_ = "mustard";
static const std::string frame_id = "1581120424100262102";


void draw3DBoundingBox(const Eigen::Matrix3f& intrinsic, 
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

Eigen::Matrix3f ReadCamK(const std::string& cam_K_path)
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

TEST(foundationpose_test, test) 
{
  auto refiner_core = CreateTrtInferCore(refiner_engine_path_,
                                          {
                                            {"transf_input", {252, 160, 160, 6}},
                                            {"render_input", {252, 160, 160, 6}},
                                          },
                                          {
                                            {"trans", {252, 3}},
                                            {"rot", {252, 3}}
                                          }, 
                                          1);
  auto scorer_core = CreateTrtInferCore(scorer_engine_path_,
                                        {
                                          {"transf_input", {252, 160, 160, 6}},
                                          {"render_input", {252, 160, 160, 6}},
                                        },
                                        {
                                          {"scores", {252, 1}}
                                        },
                                        1);
  
  Eigen::Matrix3f intrinsic_in_mat = ReadCamK(demo_data_path_ + "/cam_K.txt");

  auto foundation_pose = CreateFoundationPoseModel(refiner_core, 
                                                  scorer_core,
                                                  demo_name_,
                                                  demo_textured_obj_path,
                                                  demo_textured_map_path,
                                                  intrinsic_in_mat);
                                                

  cv::Mat rgb = cv::imread(demo_data_path_ + "/rgb/" + frame_id + ".png");
  cv::Mat depth = cv::imread(demo_data_path_ + "/depth/" + frame_id + ".png", cv::IMREAD_UNCHANGED);
  cv::Mat mask = cv::imread(demo_data_path_ + "/masks/" + frame_id + ".png", cv::IMREAD_UNCHANGED);

  depth.convertTo(depth, CV_32FC1);
  depth = depth / 1000.f;

  cv::cvtColor(rgb, rgb, cv::COLOR_BGR2RGB);
  cv::cvtColor(mask, mask, cv::COLOR_BGR2RGB);
  std::vector<cv::Mat> channels;
  cv::split(mask, channels);

  mask = channels[0];

  const Eigen::Vector3f object_dimension = foundation_pose->GetObjectDimension(demo_name_);
  
  Eigen::Matrix4f out_pose;
  foundation_pose->Register(rgb.clone(), depth, mask, demo_name_, out_pose);

  // [temp] for test
  cv::Mat regist_plot = rgb.clone();
  draw3DBoundingBox(intrinsic_in_mat, out_pose, 480, 640, object_dimension, regist_plot);
  cv::imwrite("/workspace/test_data/test_foundationpose_plot.png", regist_plot);

  Eigen::Matrix4f track_pose;
  foundation_pose->Track(rgb.clone(), depth, demo_name_, track_pose);
  cv::Mat track_plot = rgb.clone();
  draw3DBoundingBox(intrinsic_in_mat, track_pose, 480, 640, object_dimension, track_plot);
  cv::imwrite("/workspace/test_data/test_foundationpose_track_plot.png", track_plot);
}



TEST(foundationpose_test, speed_register) 
{
  auto refiner_core = CreateTrtInferCore(refiner_engine_path_,
                                          {
                                            {"transf_input", {252, 160, 160, 6}},
                                            {"render_input", {252, 160, 160, 6}},
                                          },
                                          {
                                            {"trans", {252, 3}},
                                            {"rot", {252, 3}}
                                          }, 
                                          1);
  auto scorer_core = CreateTrtInferCore(scorer_engine_path_,
                                        {
                                          {"transf_input", {252, 160, 160, 6}},
                                          {"render_input", {252, 160, 160, 6}},
                                        },
                                        {
                                          {"scores", {252, 1}}
                                        },
                                        1);
  
  Eigen::Matrix3f intrinsic_in_mat = ReadCamK(demo_data_path_ + "/cam_K.txt");

  auto foundation_pose = CreateFoundationPoseModel(refiner_core, 
                                                  scorer_core,
                                                  demo_name_,
                                                  demo_textured_obj_path,
                                                  demo_textured_map_path,
                                                  intrinsic_in_mat);
                                                

  cv::Mat rgb = cv::imread(demo_data_path_ + "/rgb/" + frame_id + ".png");
  cv::Mat depth = cv::imread(demo_data_path_ + "/depth/" + frame_id + ".png", cv::IMREAD_UNCHANGED);
  cv::Mat mask = cv::imread(demo_data_path_ + "/masks/" + frame_id + ".png", cv::IMREAD_UNCHANGED);

  depth.convertTo(depth, CV_32FC1);
  depth = depth / 1000.f;

  cv::cvtColor(rgb, rgb, cv::COLOR_BGR2RGB);
  cv::cvtColor(mask, mask, cv::COLOR_BGR2RGB);
  std::vector<cv::Mat> channels;
  cv::split(mask, channels);

  mask = channels[0];

  const Eigen::Vector3f object_dimension = foundation_pose->GetObjectDimension(demo_name_);

  // proccess
  FPSCounter counter;
  counter.Start();
  for (int i = 0 ; i < 50 ; ++ i) {
    Eigen::Matrix4f out_pose;
    foundation_pose->Register(rgb.clone(), depth, mask, demo_name_, out_pose);
    counter.Count(1);
  }

  LOG(WARNING) << "average fps: " << counter.GetFPS();

  
}



TEST(foundationpose_test, speed_track) 
{
  auto refiner_core = CreateTrtInferCore(refiner_engine_path_,
                                          {
                                            {"transf_input", {252, 160, 160, 6}},
                                            {"render_input", {252, 160, 160, 6}},
                                          },
                                          {
                                            {"trans", {252, 3}},
                                            {"rot", {252, 3}}
                                          }, 
                                          1);
  auto scorer_core = CreateTrtInferCore(scorer_engine_path_,
                                        {
                                          {"transf_input", {252, 160, 160, 6}},
                                          {"render_input", {252, 160, 160, 6}},
                                        },
                                        {
                                          {"scores", {252, 1}}
                                        },
                                        1);
  
  Eigen::Matrix3f intrinsic_in_mat = ReadCamK(demo_data_path_ + "/cam_K.txt");

  auto foundation_pose = CreateFoundationPoseModel(refiner_core, 
                                                  scorer_core,
                                                  demo_name_,
                                                  demo_textured_obj_path,
                                                  demo_textured_map_path,
                                                  intrinsic_in_mat);
                                                

  cv::Mat rgb = cv::imread(demo_data_path_ + "/rgb/" + frame_id + ".png");
  cv::Mat depth = cv::imread(demo_data_path_ + "/depth/" + frame_id + ".png", cv::IMREAD_UNCHANGED);
  cv::Mat mask = cv::imread(demo_data_path_ + "/masks/" + frame_id + ".png", cv::IMREAD_UNCHANGED);

  depth.convertTo(depth, CV_32FC1);
  depth = depth / 1000.f;

  cv::cvtColor(rgb, rgb, cv::COLOR_BGR2RGB);
  cv::cvtColor(mask, mask, cv::COLOR_BGR2RGB);
  std::vector<cv::Mat> channels;
  cv::split(mask, channels);

  mask = channels[0];

  const Eigen::Vector3f object_dimension = foundation_pose->GetObjectDimension(demo_name_);
  
  Eigen::Matrix4f out_pose;
  foundation_pose->Register(rgb.clone(), depth, mask, demo_name_, out_pose);

  // proccess
  FPSCounter counter;
  counter.Start();
  for (int i = 0 ; i < 5000 ; ++ i) {
    Eigen::Matrix4f track_pose;
    foundation_pose->Track(rgb.clone(), depth, demo_name_, track_pose);
    counter.Count(1);
  }

  LOG(WARNING) << "average fps: " << counter.GetFPS();
  
}