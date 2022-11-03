
#! /bin/bash

# run all the filters for validation set

sbatch /home/bailiping/Desktop/MOT/slurm/run_gnn_centerpoint.slurm
sleep 5
sbatch /home/bailiping/Desktop/MOT/slurm/run_gnn_megvii.slurm
sleep 5
sbatch /home/bailiping/Desktop/MOT/slurm/run_gnn_pointpillars.slurm
sleep 5
sbatch /home/bailiping/Desktop/MOT/slurm/run_phd_centerpoint.slurm
sleep 5
sbatch /home/bailiping/Desktop/MOT/slurm/run_phd_megvii.slurm
sleep 5
sbatch /home/bailiping/Desktop/MOT/slurm/run_phd_pointpillars.slurm
sleep 5
sbatch /home/bailiping/Desktop/MOT/slurm/run_pmb_centerpoint.slurm
sleep 5
sbatch /home/bailiping/Desktop/MOT/slurm/run_pmb_megvii.slurm
sleep 5
sbatch /home/bailiping/Desktop/MOT/slurm/run_pmb_pointpillars.slurm
sleep 5
sbatch /home/bailiping/Desktop/MOT/slurm/run_pmbm_centerpoint.slurm
sleep 5
sbatch /home/bailiping/Desktop/MOT/slurm/run_pmbm_megvii.slurm
sleep 5
sbatch /home/bailiping/Desktop/MOT/slurm/run_pmbm_pointpillars.slurm
sleep 5
sbatch /home/bailiping/Desktop/MOT/slurm/run_pmbmgnn_centerpoint.slurm
sleep 5
sbatch /home/bailiping/Desktop/MOT/slurm/run_pmbmgnn_megvii.slurm
sleep 5
sbatch /home/bailiping/Desktop/MOT/slurm/run_pmbmgnn_pointpillars.slurm
