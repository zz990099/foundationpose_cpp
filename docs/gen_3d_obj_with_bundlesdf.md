# gen_3d_obj_using_bundlesdf

## 1. 构建 `BundleSDF` 项目环境

1. Download `BundleSDF` repo.
```bash
  git clone https://github.com/NVlabs/BundleSDF.git
```
2. Build docker image
```bash
  cd BundleSDF/docker
  docker build --network host -t nvcr.io/nvidian/bundlesdf .
```
3. Build docker container
```bash
  bash run_container.sh
```
4. Build packages inside container
```bash
  bash build.sh
```
5. (**Optional**) 官方脚本构建的镜像可能存在一些问题，我们遇到了`libstdc++.so`库版本问题，以及opengl库缺失问题，分别解决：
```bash
  # fix the libstdc++.so version issue
  cp /opt/conda/lib/libstdc++.so.6.0.29 /usr/lib/x86_64-linux-gnu/
  rm /usr/lib/x86_64-linux-gnu/libstdc++.so.6
  ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.29 /usr/lib/x86_64-linux-gnu/libstdc++.so.6

  # fix opengl package missing issue
  pip install PyOpenGL-accelerate
```

## 2. 准备数据

`FoundationPose`算法依赖于目标物的三维模型，包括三个文件，可由`BundleSDF`等三维重建算法得到。
  1. textured_mesh.obj
  2. material.mtl
  3. meterial_0.png

直接使用公开的数据集：

由`BundleSDF`生成三维模型，步骤：
  1. 需要一个RGB-D camera (Intel Realsense D435, etc.)，并得到它的**内参**。
  2. 采集**rgb图像**和**深度图**。
  3. 由`SAM`等分割算法，得到rgb图像序列中的**目标物掩码**。
  4. 交给`BundleSDF`算法构造三维模型。

### 2.1 Realsense相机及其内参

需要安装`librealsense2`相关包，使用`realsense-viewer`的功能，参考[Linux下安装librealsense2](https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md)，或其他任意方式，只要能安装成功。

使用docker来安装librealsense2
```bash
  docker run -itd --rm --privileged --network=host \
                  --device /dev/dri \
                  --group-add video \
                  -e DISPLAY=${DISPLAY}  \
                  -v /home/${USER}:/home/${USER} \
                  -w /home/${USER} \
                  ubuntu:20.04 \
                  /bin/bash

  # in container
  sed -i 's@//.*archive.ubuntu.com@//mirrors.ustc.edu.cn@g' /etc/apt/sources.list
  apt-get update -y
  apt-get install sudo curl lsb-release -y
  sudo mkdir -p /etc/apt/keyrings
  curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | sudo tee /etc/apt/keyrings/librealsense.pgp > /dev/null
  echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo `lsb_release -cs` main" | sudo tee /etc/apt/sources.list.d/librealsense.list
  sudo apt-get update
  sudo apt-get install librealsense2-dkms
  sudo apt-get install librealsense2-utils
```

启动realsense-viewer，录制640x480像素，30fps的bag包，期间保持相机静止，手持目标物体，尽可能多地拍摄物体各个角度。

### 2.2 采集rgb图像和depth深度图

得到realsense的bag包，需要处理成rgb图像序列和depth深度图序列，python脚本:
```python
  import pyrealsense2 as rs
  import numpy as np
  import cv2
  import os

  # 创建管道
  pipeline = rs.pipeline()
  config = rs.config()

  # 打开 .bag 文件
  config.enable_device_from_file('/workspace/realsense_bags/20240912_155301.bag')

  # 开始管道
  pipeline.start(config)

  # 创建对齐对象
  align = rs.align(rs.stream.color)

  # 创建输出目录
  depth_output_dir = 'depth'
  color_output_dir = 'rgb'
  os.makedirs(depth_output_dir, exist_ok=True)
  os.makedirs(color_output_dir, exist_ok=True)

  try:
      index = 0
      while True:
          # 等待新帧
          frames = pipeline.wait_for_frames()

          # 对齐帧
          aligned_frames = align.process(frames)

          # 获取对齐后的深度和颜色帧
          depth_frame = aligned_frames.get_depth_frame()
          color_frame = aligned_frames.get_color_frame()

          if not depth_frame or not color_frame:
              continue

          # 转换为 NumPy 数组
          depth_image = np.asanyarray(depth_frame.get_data())
          color_image = np.asanyarray(color_frame.get_data())

          # 将 BGR 转换为 RGB
          color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

          # 保存图像
          cv2.imwrite(os.path.join(depth_output_dir, f'{index:05d}.png'), depth_image)
          cv2.imwrite(os.path.join(color_output_dir, f'{index:05d}.png'), color_image)

          index += 1

          # 显示图像（可选）
          cv2.imshow('Depth Image', depth_image)
          cv2.imshow('Color Image', color_image)

          # 按 'q' 键退出
          if cv2.waitKey(1) & 0xFF == ord('q'):
              break
  finally:
      # 停止管道
      pipeline.stop()
      cv2.destroyAllWindows()
```

### 2.3 生成目标物图像掩码

我们采用基于SAM的目标跟踪算法来实现自动标注目标物掩码，算法来自[XMem](https://github.com/hkchengrex/XMem).

1. 下载XMem工程
```bash
  git clone https://github.com/hkchengrex/XMem.git
```

2. 创建运行环境
```bash
  conda create -n xmem python=3.8
  pip install -r requirements.txt
  # fix some packages missing
  pip install scipy Cython PySide6 opencv-python
```

3. 下载模型
```bash
  bash scripts/download_models_demo.sh
```

4. 运行demo, 左键点击目标物，看到目标物的mask正确，再点击`Forward Propagate`按钮
```bash
  python3 interactive_demo.py --images ${repo_path}/demo_data/rgb/
```

5. 最后在`workspace`文件夹下，得到与`rgb`中图像序列文件名对应的`masks`图像掩码。


******至此得到了`BundleSDF`需要的所有数据******
- 数据示例：
```bash
  demo_data
  ├── cam_K.txt
  ├── depth
  ├── masks
  └── rgb
```


## 3. 使用BundleSDF构建三维模型

运行第一步，最后可能会报错，似乎不用管他:
```bash
  python run_custom.py --mode run_video --video_dir /home/${USER}/projects/BundleSDF/demo_data --out_folder /home/${USER}/projects/BundleSDF/demo_result --use_segmenter 0 --use_gui 0 --debug_level 1
```

运行第二步，生成三维模型:
```bash
  python run_custom.py --mode global_refine --video_dir /home/${USER}/projects/BundleSDF/demo_data --out_folder /home/${USER}/projects/BundleSDF/demo_result
```
