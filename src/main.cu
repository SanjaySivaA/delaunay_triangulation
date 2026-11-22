#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <iostream>
#include <random>
#include <cmath>

// Compute 2D grid cell index for each point
__global__ void compute_grid_keys(const float2* points, int* cell_keys, int n, int grid_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float2 p = points[idx];
    int gx = min(max(int(p.x * grid_dim),0),grid_dim - 1);
    int gy = min(max(int(p.y * grid_dim),0),grid_dim - 1);
    cell_keys[idx] = gy * grid_dim + gx;
}

// Initialize the bounding super-triangle
__global__ void init_super_triangle(float2* points, int3* triangles, int3* neighbors, int n) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    points[n] = make_float2(-10.0f,-10.0f);
    points[n + 1] = make_float2(10.0f,-10.0f);
    points[n + 2] = make_float2(0.0f, 10.0f);

    triangles[0] = make_int3(n,n + 1,n + 2);
    neighbors[0] = make_int3(-1,-1,-1);
}

int main() {
    const int N = 10000;               
    const int max_triangles = 2 * N + 1;
    const int grid_dim = static_cast<int>(std::sqrt(N));

    thrust::host_vector<float2> h_points(N + 3);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.05f,0.95f);
    for (int i = 0; i < N; ++i) {
        h_points[i] = make_float2(dist(rng),dist(rng));
    }

    thrust::device_vector<float2> d_points = h_points;
    thrust::device_vector<int> d_cell_keys(N);
    thrust::device_vector<int3> d_triangles(max_triangles);
    thrust::device_vector<int3> d_neighbors(max_triangles);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    compute_grid_keys<<<blocks, threads>>>(
        thrust::raw_pointer_cast(d_points.data()),
        thrust::raw_pointer_cast(d_cell_keys.data()),
        N, grid_dim
    );
    cudaDeviceSynchronize();

    thrust::sort_by_key(d_cell_keys.begin(),d_cell_keys.end(),d_points.begin());

    init_super_triangle<<<1, 1>>>(
        thrust::raw_pointer_cast(d_points.data()),
        thrust::raw_pointer_cast(d_triangles.data()),
        thrust::raw_pointer_cast(d_neighbors.data()),
        N
    );
    cudaDeviceSynchronize();

    std::cout << "Initialized " << N << " spatially sorted points and allocated "<< max_triangles << " triangle slots on GPU\n";
    return 0;
}
