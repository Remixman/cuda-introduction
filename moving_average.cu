#include <stdio.h>
#include <math.h>

#define N 5000000
#define BLOCKSIZE 256

__global__ void moving_average(float *in, float *out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 1 && i < N-1) {
    out[i] = (in[i-1] + in[i] + in[i+1]) / 3.0;
  }
}

int main() {
  
  float *in, *out;
  float *d_in, *d_out;
  size_t vecSize = N * sizeof(float);

  in = (float*)malloc(vecSize);
  out = (float*)malloc(vecSize);

  // Allocate device memory for vector a, b and c
  cudaMalloc((void**)&d_in, vecSize);
  cudaMalloc((void**)&d_out, vecSize);

  // Transfer data from host to device
  cudaMemcpy(d_in, in, vecSize, cudaMemcpyHostToDevice);

  // Call kernel
  int threadsPerBlock = BLOCKSIZE;
  int numBlocks = ceil(N * 1.0 / threadsPerBlock);
  moving_average<<<numBlocks, threadsPerBlock>>>(d_in, d_out);

  // Transfer data from device to host
  cudaMemcpy(out, d_out, vecSize, cudaMemcpyDeviceToHost);

  // Deallocate device memory
  cudaFree(d_in);
  cudaFree(d_out);

  free(in); free(out);
  
  return 0;
}