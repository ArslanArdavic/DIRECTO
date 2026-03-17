# ── Base image ────────────────────────────────────────────────────────────────
# NVIDIA's official CUDA 12.1 image on Ubuntu 22.04 with the full development
# toolkit. No conda, no venv — packages go directly into the system Python.
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ── System dependencies and graph-tool ────────────────────────────────────────
# Everything that cannot come from pip is handled here in one RUN block.
# Combining all apt operations into one block avoids cache staleness issues
# where Docker might reuse a stale package list with newer install commands.
#
# graph-tool is a compiled C++ library with no PyPI distribution. Its authors
# publish pre-compiled Debian packages at downloads.skewed.de. Ubuntu 22.04
# (jammy) is the target platform and Python 3.10 is its default interpreter —
# graph-tool's apt bindings are compiled against exactly this combination.
RUN apt-get update && apt-get install -y \
    wget curl git build-essential \
    python3.10 python3.10-dev python3-pip \
    && wget -O /etc/apt/trusted.gpg.d/graph-tool.gpg \
       "https://downloads.skewed.de/apt/conf/graph-tool.gpg" \
    && echo "deb [signed-by=/etc/apt/trusted.gpg.d/graph-tool.gpg] https://downloads.skewed.de/apt jammy main" \
       > /etc/apt/sources.list.d/graph-tool.list \
    && apt-get update && apt-get install -y python3-graph-tool \
    && rm -rf /var/lib/apt/lists/*

# ── Copy requirements before source code ──────────────────────────────────────
# Copying requirements.txt before the full source exploits Docker's layer cache.
# The expensive pip install below only re-runs when requirements.txt changes.
# A commit that only touches Python source will skip straight past it.
WORKDIR /workspace
COPY requirements.txt .

# ── Python packages via pip ────────────────────────────────────────────────────
# We install directly into the system Python with no venv. The container is
# already a fully isolated environment at the filesystem level — a venv inside
# Docker would be solving a problem Docker has already solved upstream.
#
# --break-system-packages tells pip to write into the system Python's
# site-packages despite it being technically apt-managed. This is intentional
# and safe: this container has one purpose and one project, nothing to protect.
#
# PyTorch and DGL do not publish on PyPI — they distribute CUDA-specific wheels
# on their own servers. The --extra-index-url flags tell pip where to look for
# them. The DGL URL encodes both the CUDA version (cu121) and the PyTorch
# version (torch-2.4), ensuring binary compatibility with our PyTorch build.
RUN pip install --break-system-packages --upgrade pip && \
    pip install --break-system-packages \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
        --extra-index-url https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html \
        -r requirements.txt

# ── Copy source and install the package in editable mode ──────────────────────
# Source is copied after the dependency layer so code changes don't invalidate
# the expensive pip install above.
#
# Problem 2+3 fix: pip install -e . registers the package in editable mode,
# pointing Python at /workspace/src so all internal imports resolve correctly.
COPY . .
RUN pip install --break-system-packages -e .

# ── Build-time verification ────────────────────────────────────────────────────
# Import checks run at build time so environment problems surface during
# 'docker build' rather than silently at job submission time on the cluster.
RUN python3 -c "import torch; print('torch', torch.__version__)"
RUN python3 -c "import dgl; print('dgl', dgl.__version__)"
RUN python3 -c "import graph_tool; print('graph_tool', graph_tool.__version__)"
RUN python3 -c "import pytorch_lightning; print('lightning', pytorch_lightning.__version__)"
RUN python3 -c "import numpy; print('numpy', numpy.__version__)"

# ── Runtime working directory ──────────────────────────────────────────────────
# main.py must run from inside src/ because Hydra resolves config paths
# relative to the working directory at launch time.
WORKDIR /workspace/src

CMD ["bash"]