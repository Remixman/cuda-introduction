#include <stdio.h>
#include <math.h>

#define N 5000000

__global__ void vecadd(float *a, float *b, float *c) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) c[i] = a[i] + b[i];
}

int main() {
  
  float *a, *b, *c;
  float *d_a, *d_b, *d_c;
  size_t vecSize = N * sizeof(float);

  a = (float*)malloc(vecSize);
  b = (float*)malloc(vecSize);
  c = (float*)malloc(vecSize);

  // Allocate device memory for vector a, b and c
  cudaMalloc((void**)&d_a, vecSize);
  cudaMalloc((void**)&d_b, vecSize);
  cudaMalloc((void**)&d_c, vecSize);

  // Transfer data from host to device
  cudaMemcpy(d_a, a, vecSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, vecSize, cudaMemcpyHostToDevice);

  // Call kernel
  int threadsPerBlock = 256;
  int numBlocks = ceil(N * 1.0 / threadsPerBlock);
  vecadd<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c);

  // Transfer data from device to host
  cudaMemcpy(c, d_c, vecSize, cudaMemcpyDeviceToHost);

  // Deallocate device memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  free(a); free(b); free(c);
  
  return 0;
}