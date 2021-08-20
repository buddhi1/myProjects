#!/bin/bash
#SBATCH -N 2
#SBATCH -p GPU
#SBATCH -t 1:00:00
#SBATCH --gpus=8

# echo commands to stdout
set -x

mpicc -c main.c -o main.o
nvcc -arch=sm_72 -c multiply.cu -o multiply.o
nvcc -w -m64 -gencode arch=compute_72,code=sm_72 -gencode arch=compute_70,code=sm_70 -c multiply.cu -o multiply.o

mpicc main.o multiply.o -lcudart -L/jet/packages/spack/opt/spack/linux-centos8-zen/gcc-8.3.1/cuda-11.1.1-a6ajxenobex5bvpejykhtnfut4arfpwh/lib64 -o program

mpiexec -np 2 ./program