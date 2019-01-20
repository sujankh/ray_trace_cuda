#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <cmath>

#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>


#include "vec3.h"

// Mathematics obtained from Fundamentals of Computer Graphics (3rd edition) book

struct sphere;
struct scene;

struct vec2
{
    int x, y;
};

struct material
{
    vec3 color;
    // TODO: properties like phong exponent, specular coeffecient etc
};

struct light_source
{
    vec3 position;
    vec3 intensity;
};

struct image_plane
{
    int l, r; // Left, right edges of the image plane in 3D world
    int t, b; // Top, bottom edges of the image plane
    int nx, ny; // The dimension in pixels of the plane
    float distance; //Distance from the camera to the image_plane
};

struct hit_info
{
    sphere *obj;
    float t;
    vec3 hit_point;  // Point in surface of obj where the ray hits
};


struct scene
{
    // The world is a list of spheres for now
    // There are some issues in CUDA with polymorphic types
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#virtual-functions
    // Refer to branch renderable_abstract_class for more details
    sphere* world;
    int num_objects;
    vec3 background;

    image_plane image;
    vec3 camera;
    light_source light;
    light_source ambient;
};


struct sphere
{
    sphere(vec3 c, float radius): center(c), R(radius)
    {
    }

    vec3 center;
    float R; // radius
    material m;

    __device__ bool hits(vec3 d, scene* scene, hit_info* hit)
    {
        // Value of t for a parametric representation of the ray p(t) = e + td
        // where vectors e = camera, d = ray

        // Intersection of ray with sphere: c = center of sphere
        // t` = -d . (e-c) +- sqrt((d.(e-c))^2 - (d.d) ((e-c).(e-c) - R*2))
        // t = t`/(d.d)

        vec3 ce = scene->camera - center;
        float d_d = dot(d, d);
        float d_ce = dot(d, ce);

        float discriminant = d_ce * d_ce - d_d * (dot(ce, ce) - R * R);
        if(discriminant >= 0)
        {
            // Update hit_info
            hit->obj = this;

            float discriminant_sqrt = std::sqrt(discriminant);
            // The tracer only cares about the point at which the ray enters the sphere
            // So just set t as the smallest t obtained
            hit->t = fminf((-1 * d_ce + discriminant_sqrt)/d_d, (-1 * d_ce - discriminant_sqrt)/d_d);
            return true;
        }
        else
        {
            return false;
        }
    }

    __device__ vec3 normal(vec3 point)
    {
        // Expecting point to always be on the surface of the sphere
        return (point - center) / R;
    }

};


__device__ vec3 ray_at_pixel(int i, int j, image_plane &image)
{
    float u = image.l + (image.r - image.l) * (i + 0.5)/image.nx;
    float v = image.b + (image.t - image.b) * (j + 0.5)/image.ny;

    // Ray from camera towards the pixel (negative w for the direction)
    // Right handed co-ordinate system u,v,w
    vec3 ray = vec3(u, v, -1 * image.distance);
    ray.make_unit_vector();
    return ray;
}

__device__ vec3 surface_color(scene *scene, hit_info *hit)
{
    vec3 normal = hit->obj->normal(hit->hit_point);
    vec3 light_ray = unit_vector(scene->light.position - hit->hit_point); // vec(AB) = B - A

    // Use ambient + lambertian shading model: L = ka * Ia + kd * I * max(0, n.l)
    vec3 color = scene->ambient.intensity * hit->obj->m.color + hit->obj->m.color * scene->light.intensity * fmaxf(0, dot(normal, light_ray));
    return color;
}

__global__ void compute_pixel_color(vec2* data, vec3* frame_buffer, scene *sc)
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

        float t = 1000000000;
        hit_info nearest_hit;
        bool any_hit = false;

        // Check if the ray from the pixel hits any objects in the world
        for (int k = 0; k < sc->num_objects; k++)
        {
            sphere* obj= &sc->world[k];
            hit_info hit;

            // Get the nearest object which the ray p(t) = e + td hits
            // The smaller the value of t, the nearest the object is to the image_plane
            if(obj->hits(ray, sc, &hit))
            {
                if(hit.t < t)
                {
                    t = hit.t;
                    nearest_hit = hit;
                    any_hit = true;
                }
            }
        }

        if(any_hit)
        {
            nearest_hit.hit_point = sc->camera + t * ray; // p(t) = e + t*d  d = ray, e=camera
            pix_color = surface_color(sc, &nearest_hit);
        }
        frame_buffer[index] = pix_color;
    }
    else
    {
        printf("Index %d is out of bounds", index);
    }
}


// Unary reduction op to get the max value of either r/g/b
struct max_color
{
    __host__ __device__
    float operator()(const vec3& v)
    {
        return thrust::max(thrust::max(v.r(), v.g()), v.b());
    }

};


// Normalizer to normalize samples down to [0, 1] and then scale
struct normalize_color
{
    const float normalizer, scale;

    normalize_color(float _normalizer, int _scale): normalizer(_normalizer), scale(_scale) {}

    int operator()(const float &val) const
    {
        return static_cast<int>(val/normalizer * scale);
    }
};

int main(void)
{
    int nx = 1024;
    int ny = 1024;

    int num_pixels = nx * ny;
    dim3 threads_per_block(16, 16);
    dim3 num_blocks(nx/threads_per_block.x, ny/threads_per_block.y);

    vec3* frame_buffer;
    vec2* data;
    cudaMallocManaged(&frame_buffer, num_pixels * sizeof(vec3));
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

    int num_objects = 3;
    sphere* spheres;
    cudaMallocManaged(&spheres, num_objects * sizeof(sphere));

    spheres[0] = sphere(vec3(-4, 0, -10), 1.2);
    spheres[0].m.color = vec3(0.9, 0, 0);

    spheres[1] = sphere(vec3(3, 0, -10), 1.4);
    spheres[1].m.color = vec3(0, 0.9, 0);

    // This sphere is slightly hidden behind the 2nd sphere
    spheres[2] = sphere(vec3(2, 0, -12), 1.5);
    spheres[2].m.color = vec3(0, 0, 0.9);

    image_plane image;
    image.l = -4; image.r = 4;
    image.t = 4; image.b = -4;
    image.nx = nx; image.ny = ny;
    image.distance = 4;

    scene *sc;
    cudaMallocManaged(&sc, sizeof(scene));
    sc->background = vec3(0.4, 0.4, 0.4);
    sc->world = spheres;
    sc->num_objects = num_objects;
    sc->camera = vec3(0, 0, 0);
    sc->image = image;

    sc->light.position = vec3(-6, 4, -4);
    sc->light.intensity = vec3(1, 1, 1);
    sc->ambient.intensity = vec3(0.3, 0.3, 0.3);

    compute_pixel_color<<<num_blocks, threads_per_block>>>(data, frame_buffer, sc);
    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();
    if(error!=cudaSuccess)
    {
        fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
        return 0;
    }

    // Scale all color values to [0, 1]
    thrust::device_ptr<vec3> dev_frame_buffer_begin(frame_buffer), dev_frame_buffer_end(frame_buffer + num_pixels);
    float normalizer = thrust::transform_reduce(dev_frame_buffer_begin, dev_frame_buffer_end, max_color(), 0.0f, thrust::maximum<float>());

    int max_color_value = (2 << 15) - 1;  // The max color value for a PPM file
    normalize_color normalize(normalizer, max_color_value);

    std::cout << "P3\n" << nx << " " << ny << "\n" << max_color_value << "\n";
    for (int i = 0; i < num_pixels; ++i)
    {
        std::cout << normalize(frame_buffer[i].r()) << " " << normalize(frame_buffer[i].g()) << " " << normalize(frame_buffer[i].b()) << "\n";
    }

    cudaFree(spheres);
    cudaFree(sc);
    cudaFree(data);
    cudaFree(frame_buffer);

    return 0;
}
