#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return (double) tp.tv_sec + (double)tp.tv_usec*1e-6;
}

#define CLOCK_RATE 1695000  /* Modify from below */
__device__ void sleep(float t) {    
    clock_t t0 = clock64();
    clock_t t1 = t0;
    while ((t1 - t0)/(CLOCK_RATE*1000.0f) < t)
        t1 = clock64();
}

__global__ void mykernel() {
    sleep(1.0);    
}

int main(int argc, char* argv[]) {
    cudaDeviceProp  prop;
    cudaGetDeviceProperties(&prop, 0); 
    clock_t clock_rate = prop.clockRate;

    int num_blocks = atoi(argv[1]);

    dim3 block(1);
    dim3 grid(num_blocks);  /* N blocks */

    double start = cpuSecond();
    mykernel<<<grid,block>>>();
    cudaDeviceSynchronize();
    double etime = cpuSecond() - start;

    printf("clock_rate          %10d\n", clock_rate);
    printf("time                %10.2f\n", etime);

    cudaDeviceReset();
}
