#!/bin/bash
####################################
#        GPU Job         #
####################################
#SBATCH --job-name=unet_segmentation              # Job name
#SBATCH --output=../logs/train_segmentation.%j.out   # Stdout (%j expands to jobId)
#SBATCH --error=../logs/train_segmentation.%j.err    # Stderr (%j expands to jobId)
#SBATCH --partition=ml           # instead of comute
#SBATCH --mem=240G               # memory per NODE
#SBATCH --gres=gpu:4             # to reserve 1 of the 2 available gpu
#SBATCH --nodes=1                # to reserve 1 node
#SBATCH --ntasks=1               # to reserve half of cpu cores that correspong to # 1 of the 2 available gpus
#SBATCH --cpus-per-task=44       # Threads per task(=1) for pure MPI
#SBATCH --time=12:00:00          # Walltime
#SBATCH -A pa171004              # Accounting project

export OMP_NUM_THREADS=1

module load swbwl
module load openblas
module load gnu/6.4.0
module load java/1.8.0
module load cuda/9.2.148
module load intelmpi/2018.3
module load intel/18.0.3
module load tensorflow/1.10.1gpu


srun python myTrain.py
