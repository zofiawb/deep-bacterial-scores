#!/bin/bash
# SLURM script for train_cnn_rinalmo_initial.py

#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --job-name="training_Cnn_rinalmo_initial"
#SBATCH --account="punim1021"

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8        # feeds 4 DataLoader workers with headroom

#SBATCH --gres=gpu:1             # 1 GPU; code only uses cuda:0

#SBATCH --mem=40G                # 13M rows in RAM + micro model + batch=256

#SBATCH --mail-user=zwitkowskibl@student.unimelb.edu.au
#SBATCH --mail-type=FAIL,BEGIN,END

#SBATCH --time=0-4:0:00         # 4 hours estimate

if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

module purge
module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda activate /home/zwitkowskibl/.conda/envs/rinalmo

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"
echo "Working dir: $(pwd)"
nvidia-smi

python3 train_cnn_rinalmo_initial.py

echo "End: $(date)"

##DO NOT ADD/EDIT BEYOND THIS LINE##
my-job-stats -a -n -s
