#include <stdio.h>

#define N 10

__global__ void test_kernel(int *a, int *b, int *c, int device_id) {
        int tid = blockIdx.x*blockDim.x + threadIdx.x;
        // printf("here");
        if(tid < N) {
                c[tid] = a[tid] + b[tid];
                // printf("c: %d ", c[tid]);
        }
        // printf("Hello From GPU device id %u %d\n", device_id, threadIdx.x);
}

int main() {
        // int nDevices;

        // cudaGetDeviceCount(&nDevices);
        // for (int i = 0; i < nDevices; i++) {
        //         cudaDeviceProp prop;
        //         cudaGetDeviceProperties(&prop, i);
        //         printf("Device Number: %d\n", i);
        //         printf("  Device name: %s\n", prop.name);
        //         printf("  Memory Clock Rate (KHz): %d\n",
        //         prop.memoryClockRate);
        //         printf("  Memory Bus Width (bits): %d\n",
        //         prop.memoryBusWidth);
        //         printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
        //         2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
        // }

        int devices_count;
        int a[N], b[N], c[N];
        int *dev_a, *dev_b, *dev_c;
        cudaGetDeviceCount(&devices_count);

        printf("count %d\n", devices_count);

        // populate data
        for(int i=0; i<N; ++i){
                a[i]=i;
                b[i]=i;
        } 

        for (unsigned int device_id = 0; device_id < 2; device_id++)
        {
                cudaSetDevice (device_id);
                // const unsigned int chunk_size = chunk_ends[device_id] - chunk_begins[device_id];
                // const unsigned char *host_chunk_input = img->data.get () + chunk_begins[device_id];
                // unsigned char *host_chunk_output = host_result + chunk_begins[device_id];
                // cudaMemcpy (devices_inputs[device_id], host_chunk_input, chunk_size * sizeof (unsigned char), cudaMemcpyHostToDevice);
                
                // gpu_div_kernel_vec<div> <<<block_sizes[device_id], threads_per_block>>> (devices_inputs[device_id], devices_outputs[device_id]);
                
                // cudaMemcpy (host_chunk_output, devices_outputs[device_id], chunk_size * sizeof (unsigned char), cudaMemcpyDeviceToHost);

                cudaMalloc((void**)&dev_a, sizeof(int)*N);
                cudaMalloc((void**)&dev_b, sizeof(int)*N);
                cudaMalloc((void**)&dev_c, sizeof(int)*N);
                cudaMemcpy(dev_a, a, sizeof(int)*N, cudaMemcpyHostToDevice);
                cudaMemcpy(dev_b, b, sizeof(int)*N, cudaMemcpyHostToDevice);
                test_kernel<<<1, N>>>(dev_a, dev_b, dev_c, device_id);
                cudaMemcpy(dev_c, c, sizeof(int)*N, cudaMemcpyDeviceToHost);

                // for(int i=0; i<N; ++i) {
                //         printf("%d ", c[i]);
                // }
                // printf("\n");
        }

    return 0;
}