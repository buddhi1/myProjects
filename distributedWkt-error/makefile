GCC = g++
CC = mpicxx

LIBB = -L/usr/lib64 -lgeos_c
# LIBB = 
# LIBCUDA = -L/usr/local/cuda/lib64
# LIBCUDA = /jet/packages/spack/opt/spack/linux-centos8-zen/gcc-8.3.1/cuda-10.2.89-kz7u4ix6ed53nioz4ycqin3kujcim3bs/bin/nvcc
LIBCUDA = -L/jet/packages/pgi/21.5/Linux_x86_64/21.5/cuda/lib64
# LIBCUDA =

LFLAGS = -Wall
LFLAGS = 
LIBRA = 
DEBUG =
# CFLAGS = -m64 -O2 -Wall -c -std=c++0x -I/usr/local/include
CFLAGS = -m64 -O2 -Wall -c -std=c++0x -I/usr/include
# CFLAGS = -m64 -O2 -Wall -c -I/jet/packages/pgi/21.5/Linux_x86_64/21.5/cuda/include

mpicuda: main.o capi_read.o
	$(CC) -O2 $(LFLAGS) -o program main.o capi_read.o $(LIBB) $(LIBRA) $(LIBCUDA) -lcudart

capi_read.o: capi_read.cu
	nvcc -w -m64 -gencode arch=compute_72,code=sm_72 -gencode arch=compute_70,code=sm_70 -o capi_read.o -c capi_read.cu 	

main.o: main.cpp
	$(CC)  $(DEBUG) $(CFLAGS) -c main.cpp