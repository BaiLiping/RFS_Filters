#! /bin/bash

#SBATCH -J pmb
#SBATCH -p cpu-high
#SBATCH -N 1
#SBATCH -n 36
#SBATCH --output=/home/zhubinglab/Desktop/nuScenes_Tracker/slurmoutfile/pmb_centerpoint.out

source activate zhubinglabenv


python3 /home/zhubinglab/Desktop/nuScenes_Tracker/run_pmb_tracker.py --detection_file='/home/zhubinglab/Desktop/mmdetection3d/data/nuscenes/official_inference_result/centerpoint_val.json'
