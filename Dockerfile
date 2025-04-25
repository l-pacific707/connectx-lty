# 1. Base Image (CUDA 11.1.1, cuDNN 8, Ubuntu 20.04)
FROM nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu20.04

# 2. Environment Variables
ENV PYTHONUNBUFFERED=1
ENV LANG=C.UTF-8
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV DEBIAN_FRONTEND=noninteractive

# 3. Install Python, pip, build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    git \
    build-essential \
    gcc \
    g++ \
    libopenblas-dev \
    liblapack-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y graphviz
RUN pip install graphviz



# 4. Python alias
RUN ln -s /usr/bin/python3 /usr/bin/python

# 5. Set workdir
WORKDIR /app

# 6. Copy and install requirements
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip wheel && \
    pip install --no-cache-dir setuptools==59.5.0 && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 \
    -f https://download.pytorch.org/whl/torch_stable.html

    # Specifically copy your scripts (or use 'COPY . .')
    COPY train.py .
    COPY MCTS_Connectx.py .
    COPY ConnectXNN.py .
    COPY logger_setup.py .


    # 7. Define Default Command
    # This command runs when the container starts, executing your training script
    CMD ["python", "train.py"]
    