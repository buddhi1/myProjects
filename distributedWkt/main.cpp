#include <iostream>
#include <mpi.h>
#include "main.h"
// #include "geoswktread.cpp"
 
int main(int argc, char *argv[])
{
        /* It's important to put this call at the begining of the program, after variable declarations. */
        MPI_Init(&argc, &argv);
        int myRank, numProcs;

        /* Get the number of MPI processes and the rank of this process. */
        MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
        MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

        // main2();

        printf("Hello World!!! from %d CPU. Total CPUs %d\n", myRank, numProcs);
        // ==== Call function 'call_me_maybe' from CUDA file multiply.cu: ==========
        call_intersection();

        /* ... */
 
}
