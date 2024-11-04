#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --account=hoepfner-np
#SBATCH --partition=hoepfner-np
#SBATCH --nodes=1
#SBATCH --ntasks=40

#SBATCH --job-name=HoomdH2O
#SBATCH -o slurm_output.std
#SBATCH -e slurm_errors.std

export WORKDIR=$HOME/Research_Code/GPFT_Publication/H2O/Simulated
export OMP_NUM_THREADS=1

module load gcc/8.5.0
module load openmpi/4.1.5-gpu
module load hoomd

echo " Started at $(date)"
echo " Using #Threads per (MPI) process:$OMP_NUM_THREADS"

cd $WORKDIR
mpirun -n 32 python3 H2O.py

echo " Ended at $(date) "