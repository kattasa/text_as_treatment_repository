#!/bin/bash
#SBATCH --job-name=conf
#SBATCH --output=slurm_outputs/array_job_%A_%a.out  # %A is the job array ID, %a is the task ID
#SBATCH --error=slurm_outputs/array_job_%A_%a.err
#SBATCH --array=0-11 # Job array range
#SBATCH --ntasks=1                    # Number of tasks per job
#SBATCH --cpus-per-task=4             # Number of CPU cores per task
#SBATCH --mem=50G                      # Memory per task
#SBATCH --time=01:00:00               # Time limit
#SBATCH --mail-type=ALL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --exclude=linux[46],linux[1]

srun -u python3 main.py --task_id $SLURM_ARRAY_TASK_ID --no_simulate --run_cb
