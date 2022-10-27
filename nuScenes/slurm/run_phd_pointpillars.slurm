#! /bin/bash

#SBATCH -J phd
#SBATCH -p cpu-high
#SBATCH -N 1
#SBATCH -n 36
#SBATCH --output=/home/zhubinglab/Desktop/nuScenes_Tracker/slurmoutfile/phd_pointpillars.out

source activate zhubinglabenv


python3 /home/zhubinglab/Desktop/nuScenes_Tracker/run_phd_tracker.py --detection_file='/home/zhubinglab/Desktop/mmdetection3d/data/nuscenes/official_inference_result/pointpillars_val.json'
