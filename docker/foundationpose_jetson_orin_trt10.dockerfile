# Base image starts with CUDA
ARG BASE_IMG=nvcr.io/nvidia/l4t-cuda:12.2.12-devel
FROM ${BASE_IMG} as base
ENV BASE_IMG=nvcr.io/nvidia/l4t-cuda:12.2.12-devel

ENV DEBIAN_FRONTEND=noninteractive

RUN rm /etc/apt/sources.list && \
    echo "deb https://mirrors.ustc.edu.cn/ubuntu-ports/ jammy main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.ustc.edu.cn/ubuntu-ports/ jammy-security main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.ustc.edu.cn/ubuntu-ports/ jammy-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.ustc.edu.cn/ubuntu-ports/ jammy-backports main restricted universe multiverse" >> /etc/apt/sources.list && \
    apt-get update

RUN apt install -y \
    build-essential \
    manpages-dev \
    wget \
    zlib1g \
    software-properties-common \
    git \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    ca-certificates \
    curl \
    llvm \
    libncurses5-dev \
    xz-utils tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    mecab-ipadic-utf8 \
    libopencv-dev \
    libeigen3-dev \
    libgtest-dev \
    libassimp-dev

# cmake
RUN cd /tmp && \
    wget https://gp.zz990099.cn/https://github.com/Kitware/CMake/releases/download/v3.22.3/cmake-3.22.3-linux-aarch64.tar.gz && \
    tar -xzvf cmake-3.22.3-linux-aarch64.tar.gz && \
    mv cmake-3.22.3-linux-aarch64 /opt/cmake-3.22.3 && \
    rm cmake-3.22.3-linux-aarch64.tar.gz && \
    ln -s /opt/cmake-3.22.3/bin/cmake /usr/local/bin/cmake

# glog
RUN cd /tmp && \
    wget https://gp.zz990099.cn/https://github.com/google/glog/archive/refs/tags/v0.7.0.tar.gz && \
    tar -xzvf v0.7.0.tar.gz && \
    cd glog-0.7.0 && \
    mkdir build && cd build && \
    cmake .. && make -j && \
    make install

# cv-cuda
RUN cd /tmp && \
    wget https://gp.zz990099.cn/https://github.com/CVCUDA/CV-CUDA/releases/download/v0.12.0-beta/cvcuda-lib-0.12.0_beta-cuda12-aarch64-linux.deb && \
    wget https://gp.zz990099.cn/https://github.com/CVCUDA/CV-CUDA/releases/download/v0.12.0-beta/cvcuda-dev-0.12.0_beta-cuda12-aarch64-linux.deb && \
    dpkg -i cvcuda-lib-0.12.0_beta-cuda12-aarch64-linux.deb && \
    dpkg -i cvcuda-dev-0.12.0_beta-cuda12-aarch64-linux.deb && \
    rm cvcuda-lib-0.12.0_beta-cuda12-aarch64-linux.deb && \
    rm cvcuda-dev-0.12.0_beta-cuda12-aarch64-linux.deb

# TensorRT
RUN cd /tmp && \
    wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.7.0/tars/TensorRT-10.7.0.23.l4t.aarch64-gnu.cuda-12.6.tar.gz && \
    tar -xzvf TensorRT-10.7.0.23.l4t.aarch64-gnu.cuda-12.6.tar.gz && \
    rm TensorRT-10.7.0.23.l4t.aarch64-gnu.cuda-12.6.tar.gz && \
    mv TensorRT-10.7.0.23 /usr/src/tensorrt && \
    cp /usr/src/tensorrt/lib/*.so* /usr/lib/aarch64-linux-gnu/ && \
    cp /usr/src/tensorrt/include/* /usr/include/aarch64-linux-gnu/
