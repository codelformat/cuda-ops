# syntax=docker/dockerfile:1
ARG CUDA_IMAGE=nvidia/cuda:13.2.1-devel-ubuntu24.04
FROM ${CUDA_IMAGE} AS dev

ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=1000
# 可选镜像源，例如 APT_MIRROR=mirrors.tuna.tsinghua.edu.cn
#                     PIP_MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple
ARG APT_MIRROR=
ARG PIP_MIRROR=
ENV DEBIAN_FRONTEND=noninteractive

RUN if [ -n "${APT_MIRROR}" ]; then \
        sed -i -E "s#https?://(archive|security)\.ubuntu\.com#http://${APT_MIRROR}#g" \
            /etc/apt/sources.list.d/ubuntu.sources; \
    fi \
 && apt-get update && apt-get install -y --no-install-recommends \
        build-essential git gdb ninja-build \
        python3 python3-venv python3-dev \
        clang-format sudo ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

# Ubuntu 24.04 基础镜像自带 UID 1000 的 ubuntu 用户，与宿主机用户冲突，先删除
RUN userdel -r ubuntu 2>/dev/null || true \
 && groupadd -g ${USER_GID} ${USERNAME} \
 && useradd -m -u ${USER_UID} -g ${USER_GID} -s /bin/bash ${USERNAME} \
 && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME}

# Python 工具链（含 torch）放在 venv 中
# torch C++/CUDA 扩展只为 sm_86 编译，缩短编译时间
ENV TORCH_CUDA_ARCH_LIST=8.6
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=${VIRTUAL_ENV}/bin:${PATH}
COPY requirements-dev.txt /tmp/requirements-dev.txt
RUN python3 -m venv ${VIRTUAL_ENV} \
 && pip install --no-cache-dir ${PIP_MIRROR:+-i ${PIP_MIRROR}} -r /tmp/requirements-dev.txt \
 && chown -R ${USER_UID}:${USER_GID} ${VIRTUAL_ENV}

USER ${USERNAME}
WORKDIR /workspace
