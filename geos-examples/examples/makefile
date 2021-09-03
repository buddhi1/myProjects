GCC = g++

# LIBB = -L/usr/lib64 -lgeos
LIBB = -L/usr/lib64 -lgeos_c
# LIBCUDA = -L/usr/local/cuda/lib64
# LIBCUDA = /jet/packages/spack/opt/spack/linux-centos8-zen/gcc-8.3.1/cuda-10.2.89-kz7u4ix6ed53nioz4ycqin3kujcim3bs/bin/nvcc
# LIBCUDA = -L/jet/packages/pgi/21.5/Linux_x86_64/21.5/cuda/lib64
LIBCUDA =

# LFLAGS = -Wall
LFLAGS = 
LIBRA = 
DEBUG =
# CFLAGS = -m64 -O2 -Wall -c -std=c++0x -I/usr/local/include
CFLAGS = -I /usr/local/include
# CFLAGS = -m64 -O2 -Wall -c -I/jet/packages/pgi/21.5/Linux_x86_64/21.5/cuda/include

read: capi_read.o
	$(GCC) -O2 $(LFLAGS) -o capi_read capi_read.o $(LIBB) $(LIBRA) $(LIBCUDA)

capi_read.o: capi_read.cpp
	$(GCC)  $(DEBUG) $(CFLAGS) -c capi_read.cpp

read_ts: capi_read_ts.o
	$(GCC) -O2 $(LFLAGS) -o capi_read_ts capi_read_ts.o $(LIBB) $(LIBRA) $(LIBCUDA)

capi_read_ts.o: capi_read_ts.cpp
	$(GCC)  $(DEBUG) $(CFLAGS) -c capi_read_ts.cpp

read_file_ts: read_file_ts.o
	$(GCC) -O2 $(LFLAGS) -o read_file_ts read_file_ts.o $(LIBB) $(LIBRA) $(LIBCUDA)

read_file_ts.o: read_file_ts.cpp
	$(GCC)  $(DEBUG) $(CFLAGS) -c read_file_ts.cpp