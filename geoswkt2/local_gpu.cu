/* multiply.cu */
#include <cuda.h>
// #include <cuda_runtime.h>
#include <stdio.h>
 
 __global__ void __multiply__ ()
 {
     printf("Hello World from GPU!!!\n");
 }
 
 extern "C" void call_me_maybe()
{
     /* ... Load CPU data into GPU buffers  */
 
     __multiply__ <<<1,2>>> ();
     cudaDeviceSynchronize();
     printf("From host cuda\n");
 
     /* ... Transfer data from GPU to CPU */
}