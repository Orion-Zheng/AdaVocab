#!/bin/bash

#SBATCH -J launch_a100_server       # Job name
#SBATCH -p long          # Queue (partition) name
#SBATCH -N 1               # Total # of nodes (must be 1 for serial)
#SBATCH -n 1               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 24:00:00        # Run time (hh:mm:ss)
#SBATCH --constraint=xgpg  # A100-40GB*1 node
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=zheng_zian@u.nus.edu
#SBATCH --mail-type=all    # Send email at begin and end of job
source ~/.bashrc
python launch_tunnel.py --type soc_gpu
