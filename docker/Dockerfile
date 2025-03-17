FROM nvcr.io/nvidia/pytorch:24.02-py3

ENV DEBIAN_FRONTEND="noninteractive" \
    MAX_JOBS=8

# Install system dependencies
RUN apt-get update \
    && apt-get install -y sudo tmux openssh-server iputils-ping net-tools iproute2 traceroute netcat \
    build-essential libopenexr-dev libxi-dev libglfw3-dev libglew-dev libomp-dev libxinerama-dev libxcursor-dev tzdata \
    && apt clean

# Install Python monitoring tools
RUN pip install py-spy pytorch_memlab

# Install correct PyTorch and FlashAttention versions
RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121 \
    && pip install flash-attn==2.7.0.post2

# Install additional Python dependencies
RUN pip install loguru tqdm ninja tensorboard \
    sentencepiece fire tabulate easydict \
    transformers==4.48.3 awscli rpyc pythonping \
    hydra-core accelerate redis opentelemetry-api \
    opentelemetry-sdk opentelemetry-exporter-otlp prometheus-client \
    omegaconf black==22.8.0 mypy-extensions pathspec tensorboardX \
    nvitop antlr4-python3-runtime==4.11.0 ray==2.40.0 \
    deepspeed==0.16.0 vllm==0.6.5 peft

# Install Java for specific dependencies
RUN apt update && apt install --fix-broken -y \
    && apt install -y default-jre-headless openjdk-8-jdk

# Install Hydra from source
RUN eval $(curl -s deploy.i.basemind.com/httpproxy) \
    && pip install git+https://github.com/facebookresearch/hydra.git

# Set NCCL configuration
COPY nccl.conf /etc/nccl.conf

# Set timezone to Asia/Shanghai
RUN apt update && apt install -y tzdata \
    && ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
    && dpkg-reconfigure --frontend noninteractive tzdata \
    && apt clean

# Fix DeepSpeed parameter offloading issues
COPY parameter_offload.py /usr/local/lib/python3.10/dist-packages/deepspeed/runtime/zero/parameter_offload.py
COPY partitioned_param_coordinator.py /usr/local/lib/python3.10/dist-packages/deepspeed/runtime/zero/partitioned_param_coordinator.py

# Set working directory
WORKDIR /workspace/