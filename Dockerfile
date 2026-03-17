# ── Base image ────────────────────────────────────────────────────────────────
# We start from graph-tool's official Docker image rather than NVIDIA's CUDA
# image. This is the most reliable way to get a working graph-tool installation
# — the image is maintained by graph-tool's own authors and already contains
# a fully functional graph-tool + Python 3.10 environment on Ubuntu 22.04.
# We then install the CUDA toolkit on top of this base, rather than trying to
# install graph-tool on top of the CUDA base (which is fragile and error-prone).
FROM tiagopeixoto/graph-tool:latest

ENV DEBIAN_FRONTEND=noninteractive

# ── CUDA 12.1 toolkit ────────────────────────────────────────────────────────
# Add NVIDIA's apt repository and install the CUDA 12.1 toolkit and cuDNN 8.
# We install the runtime libraries (not the full development toolkit) since
# pip-distributed PyTorch wheels bundle their own CUDA runtime and only need
# the driver-facing libraries to communicate with the GPU.
# The keyring package is required to authenticate NVIDIA's apt repository.
RUN apt-get update && apt-get install -y wget && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && apt-get install -y \
        cuda-toolkit-12-1 \
        libcudnn8 \
        python3-dev \
        python3-pip \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/* cuda-keyring_1.1-1_all.deb

# Set CUDA-related environment variables so PyTorch can discover the GPU
# at runtime and pip-compiled CUDA extensions know where to find the headers.
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# ── Copy requirements before source code ──────────────────────────────────────
# requirements.txt is copied before the full source to exploit Docker's layer
# cache — the expensive pip install below only re-runs when requirements.txt
# actually changes, not on every code commit.
WORKDIR /workspace
COPY requirements.txt .

# ── Python packages via pip ────────────────────────────────────────────────────
# Install directly into the system Python — no venv needed inside a container.
# PyTorch and DGL are fetched from their own CUDA-specific wheel servers rather
# than PyPI, via the --extra-index-url flags.
RUN pip install --break-system-packages --upgrade pip && \
    pip install --break-system-packages \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
        --extra-index-url https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html \
        -r requirements.txt

# ── Copy source and install in editable mode ──────────────────────────────────
COPY . .
RUN pip install --break-system-packages -e .

# ── Build-time verification ────────────────────────────────────────────────────
# These import checks run during the build so any environment problem surfaces
# as a build failure rather than a mysterious SLURM job failure later.
RUN python3 -c "import torch; print('torch', torch.__version__)"
RUN python3 -c "import dgl; print('dgl', dgl.__version__)"
RUN python3 -c "import graph_tool; print('graph_tool', graph_tool.__version__)"
RUN python3 -c "import pytorch_lightning; print('lightning', pytorch_lightning.__version__)"
RUN python3 -c "import numpy; print('numpy', numpy.__version__)"

# ── Runtime working directory ──────────────────────────────────────────────────
WORKDIR /workspace/src

CMD ["bash"]