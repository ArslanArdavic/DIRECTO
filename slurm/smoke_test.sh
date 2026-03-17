#!/bin/bash
#SBATCH --job-name=directo_smoke
#SBATCH --output=/users/arda.arslan/allab/DIRECTO/slurm/log/directo_smoke/smoke_%j.out
#SBATCH --error=/users/arda.arslan/allab/DIRECTO/slurm/log/directo_smoke/smoke_%j.err
#SBATCH --time=00:30:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G
#SBATCH --container-image ghcr.io\#arslanardavic/directo:latest
#SBATCH --container-mounts=/stratch:/stratch

echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "Start time : $(date)"

# Verify the GPU is visible inside the container before doing any real work.
# If CUDA is unavailable, sys.exit(1) causes SLURM to mark the job as FAILED
# rather than letting a silent CPU-only run complete and give misleading results.
conda run -n directo python -c "
import torch, sys
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'
print(f'PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()} | GPU: {gpu}')
if not torch.cuda.is_available():
    sys.exit(1)
"

# The Dockerfile sets WORKDIR to /workspace/src, so main.py is already the
# current directory inside the container — no cd needed.
echo "Running smoke test..."
conda run -n directo python main.py +experiment=debug

echo "Finished at: $(date)"