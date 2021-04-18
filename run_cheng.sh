#!/bin/bash
#SBATCH --gres=gpu:1              # Number of GPU(s) per node
#SBATCH --cpus-per-task=8         # CPU cores/threads
#SBATCH --mem=16G                 # memory
#SBATCH --time=2-0                # A time limit of zero requests that no time limit be imposed. Acceptable time formats include "minutes", "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds".
#SBATCH --job-name=vqa         
#SBATCH --output=logs/%j.out
#SBATCH --mail-user=cheng.chen2@mail.mcgill.ca
#SBATCH --mail-type=ALL

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# confirm gpu available
# nvidia-smi

# validate if a parameter is provided 
if [ $# -ne 1 ] 
then 
    echo "Usage $0 <filename>"
else
    # run the command
    source vqa-env/bin/activate
    python $1
fi
