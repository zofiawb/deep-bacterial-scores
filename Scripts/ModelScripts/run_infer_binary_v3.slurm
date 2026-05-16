#!/bin/bash
# SLURM script for infer_binary_v3.py
# Runs inference on the test set with the best binary v3 checkpoint
# and exports predictions CSV for benchmarking and RNAstructure input.

#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --job-name="infer_binary_v3"
#SBATCH --account="punim1021"

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

#SBATCH --gres=gpu:1

#SBATCH --mem=40G

#SBATCH --mail-user=zwitkowskibl@student.unimelb.edu.au
#SBATCH --mail-type=FAIL,BEGIN,END

#SBATCH --time=0-0:30:00          # 30 min — inference only, no training

#SBATCH --output=slurm_infer_binary_v3_%j.out
#SBATCH --error=slurm_infer_binary_v3_%j.err

if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

module purge
module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda activate /home/zwitkowskibl/.conda/envs/rinalmo

echo "=========================================="
echo "INFERENCE: binary v3 test set predictions"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"
echo "=========================================="
nvidia-smi

python3 infer_binary_v3.py

echo "End: $(date)"

##DO NOT ADD/EDIT BEYOND THIS LINE##
my-job-stats -a -n -s
