#!/bin/bash

IMAGE_BASE_NAME="foundationpose_base_dev"
BUILT_IMAGE_TAG=""

CONTAINER_NAME="foundationpose"

usage() {
  echo "Usage: $0 --platform=<platform>"
  echo "Available platforms: jetson, nvidia_gpu, rk3588"
  exit 1
}

parse_args() {
  if [ "$#" -ne 1 ]; then
    usage
  fi
  # 解析参数
  for i in "$@"; do
      case $i in
          --platform=*)
              PLATFORM="${i#*=}"
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

build_nvidia_gpu_image() {
  BUILT_IMAGE_TAG=${IMAGE_BASE_NAME}:nvidia_gpu_tensorrt_u2204
  if is_image_exist ${BUILT_IMAGE_TAG}; then
    echo Image: ${BUILT_IMAGE_TAG} exists! Skip image building process ...
  else
    docker build -f foundationpose.dockerfile -t ${BUILT_IMAGE_TAG} . 
  fi
}


add_user() {
  echo Adding User: ${USER} into container
   
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

  EXTERNAL_TAG="--runtime nvidia"

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
             ${EXTERNAL_TAG} \
             ${BUILT_IMAGE_TAG} \
             /bin/bash
}

build_nvidia_gpu_image

create_container

echo "EasyDeploy Base Dev Enviroment Built Successfully!!!"
echo "Now Run into_docker.sh"