FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    wget git build-essential \
    && rm -rf /var/lib/apt/lists/*

# Miniforge ships with mamba pre-installed and defaults to conda-forge,
# eliminating the circular problem of needing conda to install mamba.
RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
        -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p /opt/conda && \
    rm /tmp/miniforge.sh
ENV PATH="/opt/conda/bin:$PATH"

WORKDIR /workspace
COPY . .

# Problem 1 fix: remove the invalid '- e .' pip entry from environment.yaml
RUN sed -i '/^\s*- e \./d' environment.yaml

# Problem 7 fix: rewrite the NumPy pin to <2.0 before env creation so the
# environment is built correctly from the start
RUN sed -i 's/numpy==2\.3\.1/numpy<2.0/' environment.yaml

# Create the conda environment via mamba.
# --verbose ensures failures show exactly which package caused the problem.
RUN mamba env create -f environment.yaml --verbose && \
    mamba clean -afy && \
    conda clean -afy && \
    find /opt/conda -name "*.pyc" -delete && \
    find /opt/conda -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

SHELL ["conda", "run", "-n", "directo", "/bin/bash", "-c"]

# Force-reinstall PyTorch with explicit CUDA 12.1 build.
# mamba env create may silently install a CPU-only PyTorch when running on
# GPU-less hardware (e.g. GitHub Actions runners), because the solver has no
# GPU to validate against. This step overrides whatever was installed and
# guarantees the CUDA 12.1 build is present regardless of build environment.
RUN conda install -y pytorch=2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia && \
    conda clean -afy

# Assert that the CUDA build of PyTorch is actually installed.
# torch.version.cuda being None means a CPU-only build was installed.
# Failing here at build time prevents a CPU-only image from ever being
# pushed to the registry and reaching the cluster silently.
RUN python -c "
import torch
assert torch.version.cuda is not None, (
    'CPU-only PyTorch installed (torch.version.cuda=None). '
    'The conda solver chose the CPU build — check channel priorities.'
)
print(f'PyTorch {torch.__version__} compiled against CUDA {torch.version.cuda} — OK')
"

# Problem 2+3 fix: editable install
RUN pip install --upgrade pip && pip install -e .

# Problem 4 fix: install DGL explicitly from the correct channel
RUN conda install -y -c dglteam/label/th24_cu121 dgl && conda clean -afy

# Build-time verification
RUN python -c "import torch; print('torch', torch.__version__)"
RUN python -c "import dgl; print('dgl', dgl.__version__)"
RUN python -c "import graph_tool; print('graph_tool', graph_tool.__version__)"
RUN python -c "import pytorch_lightning; print('lightning', pytorch_lightning.__version__)"
RUN python -c "import numpy; print('numpy', numpy.__version__)"

WORKDIR /workspace/src

CMD ["conda", "run", "--no-capture-output", "-n", "directo", "bash"]