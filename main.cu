#include <stdio.h>
#include <iostream>
#include <cmath>

#include "vec3.h"

struct vec2
{
    int x, y;
};


__global__ void get_rgb(vec2* data, vec3* colors, int nx, int ny)
{
    int thread_row = blockIdx.y * blockDim.y + threadIdx.y;
    int thread_col = blockIdx.x * blockDim.x + threadIdx.x;

    int index = thread_row * (gridDim.x * blockDim.x) + thread_col;

    if (index < nx * ny)
    {
        float r = data[index].x / float(nx);
        float g = data[index].y / float(ny);
        float b = 0.2;

        int ir =  int(255.99 * r);
        int ig = int(255.99 * g);
        int ib = int(255.99 * b);

        colors[index] = vec3(ir, ig, ib);
    }
    else
    {
        printf("Index %d is out of bounds", index);
    }
}

int main(void)
{
    int nx = 256;
    int ny = 128;

    int num_pixels = nx * ny;
    dim3 threads_per_block(16, 16);
    dim3 num_blocks(nx/threads_per_block.x, ny/threads_per_block.y);

    vec3* colors;
    vec2* data;
    cudaMallocManaged(&colors, num_pixels * sizeof(vec3));
    cudaMallocManaged(&data, num_pixels * sizeof(data));
    for(int j = ny -1, k = 0; j >=0; j--, k++)
    {
        for(int i = 0; i < nx; i++)
        {
            int index = i + k * nx;
            data[index].x = i;
            data[index].y = j;
        }
    }

    get_rgb<<<num_blocks, threads_per_block>>>(data, colors, nx, ny);
    cudaDeviceSynchronize();
    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    for (int i = 0; i < num_pixels; ++i)
    {
        std::cout << colors[i].r() << " " << colors[i].g() << " " << colors[i].b() << "\n";
    }

    cudaFree(data);
    cudaFree(colors);

    return 0;
}
