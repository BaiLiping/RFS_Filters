#! /bin/bash

#SBATCH -J preprocess
#SBATCH -p cpu-high
#SBATCH -N 1
#SBATCH -n 36
#SBATCH --output=/home/blp/Desktop/MOT/slurmoutfile/preprocess.out

source activate zhubinglabenv


python3 /home/blp/Desktop/MOT/utils/preprocessing.py