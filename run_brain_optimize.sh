#!/bin/bash
#SBATCH --job-name=brain_optimize
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --output=brain_optimize_slurm_%j.log
#SBATCH --error=brain_optimize_slurm_%j.log

cd /home/users/shaunakm/Brain-LLM-Fine-Tuning

echo "=== Job started at $(date) ==="
echo "=== Node: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) ==="
echo "=== RAM: $(free -h | awk '/Mem/{print $2}') total ==="
echo ""

python brain_optimize.py \
    --region broca \
    --n_steps 10 \
    --n_completions 4 \
    --output_dir ./brain-optimize-output \
    --cache ./tribe-cache

echo ""
echo "=== Job completed at $(date) ==="
