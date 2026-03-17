#!/bin/bash
#SBATCH --job-name=directo_train_er_dag
#SBATCH --output=/users/arda.arslan/allab/DIRECTO/slurm/log/phase1_train/train_%j.out
#SBATCH --error=/users/arda.arslan/allab/DIRECTO/slurm/log/phase1_train/train_%j.err
#SBATCH --time=48:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G
#SBATCH --container-image ghcr.io\#arslanardavic/directo:latest
#SBATCH --container-mounts=/stratch:/stratch
#SBATCH --container-writable

echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "Phase      : 1 — Training (ER-DAG, MagLap)"
echo "Start time : $(date)"

nvidia-smi || { echo "ERROR: nvidia-smi failed — GPU not visible. Aborting."; exit 1; }


# WANDB_API_KEY is read from the environment set in ~/.bashrc on the server.
# It is injected into the container via Pyxis's environment passthrough.
# It never appears in this file and is therefore safe to commit to git.
conda run --no-capture-output -n directo \
    python main.py \
        +experiment=synthetic \
        dataset=synthetic \
        dataset.graph_type=er \
        dataset.acyclic=True \
        model.extra_features=magnetic-eigenvalues \
        general.wandb=online \
        "hydra.run.dir=/users/arda.arslan/allab/DIRECTO/outputs/\${now:%Y-%m-%d}/\${now:%H-%M-%S}-er-dag-maglap"

echo "Phase 1 finished at: $(date)"
echo "Outputs written to /stratch/arda.arslan/directo/outputs/"