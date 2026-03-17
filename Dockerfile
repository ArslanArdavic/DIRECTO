# ── Base image ────────────────────────────────────────────────────────────────
# We start from NVIDIA's official CUDA 12.1 image with cuDNN 8 and the full
# development toolkit on Ubuntu 22.04. Conda manages graph-tool, PyTorch, and
# the full Python environment — this is the most reliable path because both
# graph-tool and PyTorch publish carefully maintained conda packages that are
# guaranteed to be mutually compatible.
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ── System utilities ───────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    wget git build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Miniconda ─────────────────────────────────────────────────────────────────
# We install Miniconda because graph-tool has no PyPI distribution and is only
# reliably available as a conda-forge package. Conda also manages the PyTorch
# and CUDA libraries, ensuring all native binaries are built against compatible
# runtime versions.
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh
ENV PATH="/opt/conda/bin:$PATH"

# ── Mamba ─────────────────────────────────────────────────────────────────────
# Mamba is a drop-in replacement for conda's solver written in C++. It resolves
# large dependency graphs (like PyTorch + CUDA + graph-tool spanning multiple
# channels) dramatically faster and with lower memory usage than conda's default
# Python solver. This is critical on GitHub Actions where the runner has limited
# memory and a time budget. Without mamba, conda's solver can exhaust memory or
# time out on this environment, producing a misleading exit code 1.
RUN conda install -n base -c conda-forge mamba -y && conda clean -afy

# ── Copy project files ─────────────────────────────────────────────────────────
WORKDIR /workspace
COPY . .

# ── Problem 1 fix: remove invalid pip entry ───────────────────────────────────
# The '- e .' line in environment.yaml is not valid syntax for conda's pip
# integration. We remove it so conda env create runs without errors.
# This sed command is a no-op if the line was already removed in the repo.
RUN sed -i '/^\s*- e \./d' environment.yaml

# ── Problem 7 fix: pin NumPy before env creation ──────────────────────────────
# environment.yaml pins numpy==2.3.1, which is incompatible with PyTorch
# Lightning 2.0.4 because NumPy 2.0 removed the np.Inf alias. We rewrite the
# pin to <2.0 directly in the YAML so the environment is created correctly
# from the start rather than requiring a post-hoc downgrade.
RUN sed -i 's/numpy==2\.3\.1/numpy<2.0/' environment.yaml

# ── Create the conda environment ──────────────────────────────────────────────
# This is the heaviest step — it installs PyTorch 2.4, CUDA libraries,
# graph-tool, and all pip dependencies. Docker's layer cache means this only
# re-runs when environment.yaml changes; routine code commits skip it entirely.
# --verbose ensures that if this step fails, the log shows exactly which package
# caused the failure rather than an opaque exit code 1.
RUN mamba env create -f environment.yaml --verbose && \
    mamba clean -afy && \
    conda clean -afy && \
    find /opt/conda -name "*.pyc" -delete && \
    find /opt/conda -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Make the directo environment the default shell for subsequent RUN steps.
SHELL ["conda", "run", "-n", "directo", "/bin/bash", "-c"]

# ── Problem 2+3 fix: editable install ─────────────────────────────────────────
# Upgrade pip first — older pip versions don't support pyproject.toml editable
# installs and require a setup.py. The upgraded pip handles pyproject.toml
# correctly, registering src/ on the Python path so internal imports resolve.
RUN pip install --upgrade pip && pip install -e .

# ── Problem 4 fix: install DGL explicitly ─────────────────────────────────────
# DGL is silently omitted by conda's solver when resolving the dglteam channel.
# We install it in a separate step after the environment exists, using the
# channel tag that encodes our exact PyTorch 2.4 + CUDA 12.1 combination.
# -y suppresses the confirmation prompt that would hang a non-interactive build.
RUN conda install -y -c dglteam/label/th24_cu121 dgl && conda clean -afy

# ── Build-time verification ────────────────────────────────────────────────────
# Import checks during the build surface environment problems immediately
# rather than hours later as a mysterious SLURM job failure on the cluster.
RUN python -c "import torch; print('torch', torch.__version__)"
RUN python -c "import dgl; print('dgl', dgl.__version__)"
RUN python -c "import graph_tool; print('graph_tool', graph_tool.__version__)"
RUN python -c "import pytorch_lightning; print('lightning', pytorch_lightning.__version__)"
RUN python -c "import numpy; print('numpy', numpy.__version__)"

# ── Runtime working directory ──────────────────────────────────────────────────
# main.py must run from inside src/ because Hydra resolves config paths
# relative to the working directory at launch time.
WORKDIR /workspace/src

CMD ["conda", "run", "--no-capture-output", "-n", "directo", "bash"]