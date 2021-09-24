#!/bin/bash
#SBATCH -J MPI-GCMF-satish           # job name
#SBATCH -o output%j.out       # output file name (%j expands to jobID), this file captures standered output from the shell
#SBATCH -e output%j.err      # error file name (%j expands to jobID), this file captures standered errors genereted from the program
#SBATCH -N 3
#SBATCH -p GPU
#SBATCH -t 00:01:00
#SBATCH --gres gpu:8
#echo commands to stdout
set -x

# srun -p GPU -N 2 --gres gpu:8 -t 1:0:0 --pty bash

module addinit openmpi/4.0.5-nvhpc21.2 cuda/11.1.1

#run pre-compiled MPI program which is already in your project space
# mpirun -np 2 ./a.out
mpirun -np 2 ./gpu ../../data/64Parts/ ../../data/64Parts/

# How to run CPU GPU
# ------------------------

# make mpicuda
# sbatch run.sh 
