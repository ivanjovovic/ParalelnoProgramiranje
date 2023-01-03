#! /bin/bash
#SBATCH --job-name=gpu_test
#SBATCH --time=00:10:00
#SBATCH --ntasks=2
#SBATCH --mem-per-cpu=1024MB
#SBATCH -o matricetest-%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1


./matrica
