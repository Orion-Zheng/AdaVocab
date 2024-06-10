#!/bin/bash
#SBATCH -J SFT_full       # Job name
#SBATCH -p gpu-long          # Queue (partition) name
#SBATCH -N 2               # Total # of nodes (must be 1 for serial)
#SBATCH --ntasks-per-node=1               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 120:00:00        # Run time (hh:mm:ss)
#SBATCH --mem=784GB     # min mem per node
#SBATCH --gpus-per-node=h100-47:4
#SBATCH --mail-user=zheng_zian@u.nus.edu
#SBATCH --mail-type=all    # Send email at begin and end of job
source ~/.bashrc
bash scripts_train/soc/run_dist_train.sh \
     scripts_train/soc/experiments_config/sft_gemma-2b/gemma-2b_sft_conifg_8_h100_47.yaml