#!/bin/bash

IMAGE_BASE_NAME="foundationpose_base_dev"
BUILT_IMAGE_TAG=""

CONTAINER_TYPE=""
CONTAINER_NAME="foundationpose"

usage() {
  echo "Usage: $0 --container_type=<container_type>"
  echo "Available container_types: trt8, trt10, jetson_trt8, jetson_trt10"
  exit 1
}

parse_args() {
  if [ "$#" -ne 1 ]; then
    usage 
  fi
  # 解析参数
  for i in "$@"; do
      case $i in
          --container_type=*)
              CONTAINER_TYPE="${i#*=}"
              shift
              ;;
          *)
              usage
              ;;
      esac
  done
}

is_image_exist() {
  local name="$1"
  if docker images --filter "reference=$name" \
                   --format "{{.Repository}}:{{.Tag}}" | grep -q "$name"; then
    return 0
  else 
    return 1
  fi
}

is_container_exist() {
  local name="$1"
  if docker ps -a --filter "name=$name" | grep -q "$name"; then
    return 0
  else
    return 1
  fi
}

build_nvidia_gpu_trt8_image() {
  BUILT_IMAGE_TAG=${IMAGE_BASE_NAME}:nvidia_gpu_tensorrt8_u2204
  if is_image_exist ${BUILT_IMAGE_TAG}; then
    echo Image: ${BUILT_IMAGE_TAG} exists! Skip image building process ...
  else
    docker build -f foundationpose_nvidia_gpu_trt8.dockerfile -t ${BUILT_IMAGE_TAG} . 
  fi
}

build_nvidia_gpu_trt10_image() {
  BUILT_IMAGE_TAG=${IMAGE_BASE_NAME}:nvidia_gpu_tensorrt10_u2204
  if is_image_exist ${BUILT_IMAGE_TAG}; then
    echo Image: ${BUILT_IMAGE_TAG} exists! Skip image building process ...
  else
    docker build -f foundationpose_nvidia_gpu_trt10.dockerfile -t ${BUILT_IMAGE_TAG} . 
  fi
}

build_jetson_trt8_image() {
  BUILT_IMAGE_TAG=${IMAGE_BASE_NAME}:jetson_tensorrt8_u2204
  if is_image_exist ${BUILT_IMAGE_TAG}; then
    echo Image: ${BUILT_IMAGE_TAG} exists! Skip image building process ...
  else
    docker build -f foundationpose_jetson_orin_trt8.dockerfile -t ${BUILT_IMAGE_TAG} . 
  fi
}

build_jetson_trt10_image() {
  BUILT_IMAGE_TAG=${IMAGE_BASE_NAME}:jetson_tensorrt10_u2204
  if is_image_exist ${BUILT_IMAGE_TAG}; then
    echo Image: ${BUILT_IMAGE_TAG} exists! Skip image building process ...
  else
    docker build -f foundationpose_jetson_orin_trt10.dockerfile -t ${BUILT_IMAGE_TAG} . 
  fi
}

build_image() {
  case $CONTAINER_TYPE in
      trt8)
          echo "Start Building Docker image for nvidia_gpu trt8 ..."
          build_nvidia_gpu_trt8_image
          ;;
      trt10)
          echo "Start Building Docker image for nvidia_gpu trt10 ..."
          build_nvidia_gpu_trt10_image
          ;;
      jetson_trt8)
          echo "Start Building Docker image for jetson trt8 ..."
          build_jetson_trt8_image
          ;;
      jetson_trt10)
          echo "Start Building Docker image for jetson trt10 ..."
          build_jetson_trt10_image
          ;;
      *)
          echo "Unknown platform: $PLATFORM"
          usage
          ;;
  esac
}

create_container() {
  echo "Creating docker container ..."

  if ! is_image_exist ${BUILT_IMAGE_TAG}; then
    echo Image: ${BUILT_IMAGE_TAG} does not exist, quit creating ...
    exit 1
  fi

  if is_container_exist ${CONTAINER_NAME}; then
    echo Container: ${CONTAINER_NAME} exists! Skip container building process ...
    return 0
  fi

  docker run -itd --privileged \
             --device /dev/dri \
             --group-add video \
             -v /tmp/.X11-unix:/tmp/.X11-unix \
             --network host \
             --ipc host \
             -v $(dirname "$(pwd)"):/workspace \
             -w /workspace \
             -v /dev/bus/usb:/dev/bus/usb \
             -e DISPLAY=${DISPLAY} \
             -e DOCKER_USER=${USER} \
             -e USER=${USER} \
             --name ${CONTAINER_NAME} \
             --runtime nvidia \
             ${BUILT_IMAGE_TAG} \
             /bin/bash
}

parse_args "$@"

build_image

create_container

echo "FoundationPose Base Dev Enviroment Built Successfully!!!"
echo "Now Run into_docker.sh"