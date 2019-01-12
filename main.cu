#include <stdio.h>
#include <iostream>
#include <cmath>

#include "vec3.h"

// Mathematics obtained from Fundamentals of Computer Graphics (3rd edition) book

struct vec2
{
    int x, y;
};

struct scene;

struct sphere
{
    sphere(vec3 c, float radius): center(c), R(radius)
    {
    }

    vec3 center;
    float R; // radius

    __device__ bool hits(vec3 ray, scene* scene)
    {
        // To make the maths simple, put camera at origin
        float r_c = dot(ray, -1 * center);
        float r_r = dot(ray, ray);
        float c_c = dot(center, center);

        float discriminant = r_c * r_c - r_r * (c_c - R * R);

        if(discriminant >= 0)
        {
            return true;
        }
        else
        {
            return false;
        }
    }
};

struct image_plane
{
    int l, r; // Left, right edges of the image plane in 3D world
    int t, b; // Top, bottom edges of the image plane
    int nx, ny; // The dimension in pixels of the plane
    float distance; //Distance from the camera to the image_plane
};


struct scene
{
    sphere* world; // For now a list of spheres comprises the world
    int num_objects;
    vec3 background;

    image_plane image;
    vec3 camera;
};

__device__ vec3 ray_at_pixel(int i, int j, image_plane &image)
{
    float u = image.l + (image.r - image.l) * (i + 0.5)/image.nx;
    float v = image.b + (image.t - image.b) * (j + 0.5)/image.ny;

    // Ray from camera towards the pixel (negative w for the direction)
    // Right handed co-ordinate system u,v,w
    return vec3(u, v, -1 * image.distance);
}

__global__ void compute_pixel_color(vec2* data, vec3* colors, scene *sc)
{
    int thread_row = blockIdx.y * blockDim.y + threadIdx.y;
    int thread_col = blockIdx.x * blockDim.x + threadIdx.x;

    int index = thread_row * (gridDim.x * blockDim.x) + thread_col;

    int nx = sc->image.nx;
    int ny = sc->image.ny;

    if (index < nx * ny)
    {
        int i  = data[index].x;
        int j = data[index].y;

        vec3 ray = ray_at_pixel(i, j, sc->image);
        vec3 pix_color = sc->background;

        // Check if the ray from the pixel hits any objects in the world
        for (int k = 0; k < sc->num_objects; k++)
        {
            sphere& s= sc->world[k];
            if(s.hits(ray, sc))
            {
                // Red sphere
                pix_color = vec3(255, 0, 0);
            }
        }

        colors[index] = pix_color;
    }
    else
    {
        printf("Index %d is out of bounds", index);
    }
}

int main(void)
{
    int nx = 1024;
    int ny = 512;

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


    sphere* spheres;
    cudaMallocManaged(&spheres, 1 * sizeof(sphere));
    spheres[0] = sphere(vec3(0, 0, -10), 3);

    image_plane image;
    image.l = -2; image.r = 2;
    image.t = 3; image.b = -3;
    image.nx = nx; image.ny = ny;
    image.distance = 4;

    scene *sc;
    cudaMallocManaged(&sc, sizeof(scene));
    sc->background = vec3(255, 255, 255);
    sc->world = spheres;
    sc->num_objects = 1;
    sc->camera = vec3(0, 0, 0);
    sc->image = image;

    compute_pixel_color<<<num_blocks, threads_per_block>>>(data, colors, sc);
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
