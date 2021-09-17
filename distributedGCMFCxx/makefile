GCC = g++
CC = mpicxx

LIBB = -L/usr/lib64/ -lgeos
# LIBB = -L ~/myLibs/geos/lib/ -lgeos_c
# LIBB = 
# for Dell machine
# LIBCUDA = -L/usr/local/cuda/lib64

# LIBCUDA = /jet/packages/spack/opt/spack/linux-centos8-zen/gcc-8.3.1/cuda-10.2.89-kz7u4ix6ed53nioz4ycqin3kujcim3bs/bin/nvcc

# for bridges2
LIBCUDA = -L/jet/packages/pgi/21.5/Linux_x86_64/21.5/cuda/lib64
# LIBCUDA =

LFLAGS = -Wall
LFLAGS = 
LIBRA = 
DEBUG =
CFLAGS = -m64 -O2 -Wall -c -std=c++0x -I/usr/local/include
# CFLAGS = -m64 -O2 -Wall -c -I/jet/packages/pgi/21.5/Linux_x86_64/21.5/cuda/include

# mpicuda: main.o spatialJoin-ST_Intersect-cmbr.o geoswktRead.o
# 	$(CC) -O2 $(LFLAGS) -o program main.o spatialJoin-ST_Intersect-cmbr.o geoswktRead.o $(LIBB) $(LIBRA) $(LIBCUDA) -lcudart

# mpicuda: main.o ST_Intersect.o CudaJoinInterface.o
# 	$(CC) -O2 $(LFLAGS) -o program main.o ST_Intersect.o CudaJoinInterface.o $(LIBB) $(LIBRA) $(LIBCUDA) -lcudart

mpicuda: ST_Intersect.o CudaJoinInterface.o test.o
	$(CC) -O2 $(LFLAGS) -o program test.o ST_Intersect.o CudaJoinInterface.o $(LIBB) $(LIBRA) $(LIBCUDA) -lcudart


# local_gpu.o: local_gpu.cu
# 	nvcc -w -m64 -gencode arch=compute_72,code=sm_72 -gencode arch=compute_70,code=sm_70 -o local_gpu.o -c local_gpu.cu 	

# ST_Intersect.o: cuda_CMBR/ST_Intersect.cu
# 	nvcc -w -m64 -gencode arch=compute_72,code=sm_72 -gencode arch=compute_70,code=sm_70 -o ST_Intersect.o -c cuda_CMBR/ST_Intersect.cu 	

# *** when changing the server, first compile GPU seperately and the try make file
# for Bridges2
ST_Intersect.o: ST_Intersect/ST_Intersect.cu
	nvcc -w -m64 -gencode arch=compute_72,code=sm_72 -gencode arch=compute_70,code=sm_70 -o ST_Intersect.o -c ST_Intersect/ST_Intersect.cu 	

# for Dell machine
# ST_Intersect.o: ST_Intersect/ST_Intersect.cu
# 	nvcc -w -m64 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_60,code=sm_60 -o ST_Intersect.o -c ST_Intersect/ST_Intersect.cu 	

main.o: main.cpp
	$(CC)  $(DEBUG) $(CFLAGS) -c main.cpp

geoswktRead.o: geoswktRead.cpp
	$(CC)  $(DEBUG) $(CFLAGS) -c geoswktRead.cpp

CudaJoinInterface.o: CudaJoinInterface.cpp
	$(CC)  $(DEBUG) $(CFLAGS) -c CudaJoinInterface.cpp

test.o: test.cpp
	$(CC)  $(DEBUG) $(CFLAGS) -c test.cpp