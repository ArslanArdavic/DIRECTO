#!/bin/bash
# Run this once from the login node to submit all three phases as a chain.
# Each phase waits for the previous one to succeed before starting.
# Usage: bash slurm/submit_all.sh

REPO=/users/arda.arslan/allab/DIRECTO

JOB1=$(sbatch --parsable $REPO/slurm/phase1_train.sh)
echo "Submitted Phase 1 (training)  : job $JOB1"

# afterok means phase 2 only starts if phase 1 exits with code 0 (success).
# If phase 1 fails, phase 2 and 3 are automatically cancelled by SLURM,
# saving you from burning GPU hours on a search with no model to evaluate.
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 $REPO/slurm/phase2_search.sh)
echo "Submitted Phase 2 (search)    : job $JOB2 (waits for $JOB1)"


echo ""
echo "Monitor with : squeue -u \$USER"
echo "Phase 1 log  : tail -f $REPO/slurm/log/phase1_train/train_${JOB1}.out"
echo ""
echo "After Phase 2 completes:"
echo "  1. Read /stratch/arda.arslan/directo/outputs/.../search_hyperparameters.csv"
echo "  2. Update BEST_ETA, BEST_OMEGA, BEST_DISTORTION in slurm/phase3_eval.sh"
echo "  3. Run: sbatch $REPO/slurm/phase3_eval.sh"