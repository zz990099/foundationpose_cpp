/*
 * @Description:
 * @Author: Teddywesside 18852056629@163.com
 * @Date: 2024-11-25 14:00:38
 * @LastEditTime: 2024-11-26 22:07:03
 * @FilePath: /easy_deploy/deploy_core/include/deploy_core/common_defination.h
 */
#ifndef __EASY_DEPLOY_COMMON_DEFINATION_H
#define __EASY_DEPLOY_COMMON_DEFINATION_H

/**
 * @brief Defination of common 2D bounding box
 *
 * @param x center of bbox `x`
 * @param y center of bbox `y`
 * @param w width of bbox
 * @param h height of bbox
 * @param conf confidence of bbox
 * @param cls classification of bbox
 */
struct BBox2D {
  float x;
  float y;
  float w;
  float h;
  float conf;
  float cls;
};

/**
 * @brief Enum of data loacation
 *
 * @param HOST data is host accessable
 * @param DEVICE data is device accessable, means host cant read/write the data buffer directly
 * @param UNKOWN some other condition
 *
 */
enum DataLocation { HOST = 0, DEVICE = 1, UNKOWN = 2 };

/**
 * @brief Defination of common image format.
 *
 */
enum ImageDataFormat { YUV = 0, RGB = 1, BGR = 2, GRAY = 3 };

// some macro
#define CHECK_STATE(state, hint) \
  {                              \
    if (!(state))                \
    {                            \
      LOG(ERROR) << (hint);      \
      return false;              \
    }                            \
  }

#define MESSURE_DURATION(run)                                                                \
  {                                                                                          \
    auto start = std::chrono::high_resolution_clock::now();                                  \
    (run);                                                                                   \
    auto end = std::chrono::high_resolution_clock::now();                                    \
    LOG(INFO) << #run << " cost(us): "                                                       \
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count(); \
  }

#define MESSURE_DURATION_AND_CHECK_STATE(run, hint)                                          \
  {                                                                                          \
    auto start = std::chrono::high_resolution_clock::now();                                  \
    CHECK_STATE((run), hint);                                                                \
    auto end = std::chrono::high_resolution_clock::now();                                    \
    LOG(INFO) << #run << " cost(us): "                                                       \
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count(); \
  }

#endif