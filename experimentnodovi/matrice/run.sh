#!/bin/bash -l

#SBATCH --partition=gpu

#SBATCH -n 20
#SBATCH -N 5

#SBATCH --time=00:03:00
# SBATCH –job-name=mm


#module load OpenMPI/4.1.4-GCC-11.3.0 
module load GCCcore/10.3.0


mpirun -np 20 mpi-mm



