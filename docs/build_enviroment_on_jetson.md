# Build Enviroment On Jetson

## Jetson宿主机环境要求

`FoundationPose`算法在`TensorRT`上运行，要求`TensorRT`的版本大于等于`8.6`，所以如果在Jetson平台上使用docker构建环境，需要使用`nvcr.io/nvidia/l4t-tensorrt:r8.6.2-devel`镜像，具体镜像信息参考[nvidia ngc](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-tensorrt/tags).

而这个镜像要求宿主机的cuda版本大于等于`12.2`，所以推荐对JetsonOrin模块刷`Jetpack6.0`以上版本的镜像，参考[nvidia sdkmanager](https://developer.nvidia.com/sdk-manager).

如果使用`Jetpack5.0`作为宿主机环境，需要手动升级cuda到`12.2`版本.

## 使用docker构建环境

确保`Jetpack6.0`已经刷机成功，docker能够正常运行，直接运行项目内的构建脚本，即可生成项目运行环境。
```bash
cd ${foundationpose_cpp}/docker/
bash build_docker.sh --container_type=jetson_trt8
bash into_docker.sh
```

