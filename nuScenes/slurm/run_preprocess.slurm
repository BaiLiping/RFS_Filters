#! /bin/bash

#SBATCH -J preprocess
#SBATCH -p cpu-high
#SBATCH -N 1
#SBATCH -n 36
#SBATCH --output=/home/zhubinglab/Desktop/nuScenes_Tracker/slurmoutfile/preprocess.out

source activate zhubinglabenv


python3 /home/zhubinglab/Desktop/nuScenes_Tracker/utils/preprocessing.py