#!/bin/bash
# SLURM script for binary classifier metadata ablation study
# Ablating: Temp

#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --job-name="ablate_Temp"
#SBATCH --account="punim1021"

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

#SBATCH --gres=gpu:1

#SBATCH --mem=40G

#SBATCH --mail-user=zwitkowskibl@student.unimelb.edu.au
#SBATCH --mail-type=FAIL,BEGIN,END

#SBATCH --time=0-1:30:00

#SBATCH --output=slurm_ablate_Temp_%j.out
#SBATCH --error=slurm_ablate_Temp_%j.err

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
echo "ABLATION STUDY: --ablate Temp"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"
echo "=========================================="
nvidia-smi

python3 train_binary_ablation.py --ablate Temp

echo "End: $(date)"

##DO NOT ADD/EDIT BEYOND THIS LINE##
my-job-stats -a -n -s
