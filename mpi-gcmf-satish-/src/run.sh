#!/bin/bash
#SBATCH -J MPI-GCMF-satish           # job name
#SBATCH -o output%j.out       # output file name (%j expands to jobID), this file captures standered output from the shell
#SBATCH -e output%j.err      # error file name (%j expands to jobID), this file captures standered errors genereted from the program
#SBATCH -N 3
#SBATCH -p GPU
#SBATCH -t 00:5:00
#SBATCH --gpus=24
#echo commands to stdout
set -x

# srun -p GPU-small -N 1 --gres=gpu:8 -t 1:0:0 --pty bash
# srun -p GPU-shared -N 1 --gres=gpu:4 -t 1:0:0 --pty bash
# interact  -p GPU -N 2 --gres=gpu:8 -t 2:0:0 
# interact  -p GPU-small -N 1 --gres=gpu:8 -t 1:0:0 
# interact  -p GPU-shared -N 1 --gres=gpu:4 -t 1:0:0 
# srun -p RM-small -N 1 -t 1:0:0 --pty bash

module addinit openmpi/4.0.5-nvhpc21.2 cuda/11.1.1

#run pre-compiled MPI program which is already in your project space
# mpirun -np 2 ./a.out
# mpirun -np 1 ./gpu 2 ../../data/testData ../../data/testDataclear
# mpirun -np 1 ./gpu 2 ../../data/testData ../../data/testData

# mpirun -np 3  ./refineMC 64 ../../data/64Parts/ ../../data/64Parts/ ../../data/index\ \(mbrs\ only\)/64Parts/ ../../data/index\ \(mbrs\ only\)/64Parts/

mpirun -np 3 ./refineMC 6 ../../data/testData/ ../../data/testData/ ../../data/testDataMBRs/ ../../data/testDataMBRs/


# How to run CPU GPU
# ------------------------

# non partitioned files
# make mpicuda 
# sbatch run.sh 

# partitioned files
# make refineLB
