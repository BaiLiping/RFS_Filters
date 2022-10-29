#! /bin/bash

#SBATCH -J gnn
#SBATCH -p cpu-high
#SBATCH -N 1
#SBATCH -n 36
#SBATCH --output=/home/blp/Desktop/MOT/slurmoutfile/gnn_megvii.out

source activate zhubinglabenv


python3 /home/blp/Desktop/MOT/run_gnn_tracker.py --detection_file='/home/blp/Desktop/mmdetection3d/data/nuscenes/official_inference_result/megvii_val.json'
