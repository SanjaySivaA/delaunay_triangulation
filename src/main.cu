#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/fill.h>
#include <iostream>
#include <random>
#include <cmath>
#include <climits>

// 2D orientation test: > 0 if (a, b, c) is counter-clockwise
// Used to check if a point is inside the triangle
__device__ inline float orient2d(float2 a, float2 b, float2 c) {
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

// To check if triangle `tri` contains both vertex indices `u` and `v`
// After triangle split, the neighbours still point to the old triangle
// This function helps to figure out which of the 3 new sub-triangles now borders it and updates its neighbor pointer accordingly
__device__ inline bool has_edge(int3 tri, int u, int v) {
    bool has_u = (tri.x == u || tri.y == u || tri.z == u);
    bool has_v = (tri.x == v || tri.y == v || tri.z == v);
    return has_u && has_v;
}

// Kernel 1 : Compute 2D grid cell index for each point
__global__ void compute_grid_keys(const float2* points, int* cell_keys, int n, int grid_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float2 p = points[idx];
    int gx = min(max(int(p.x * grid_dim),0),grid_dim - 1);
    int gy = min(max(int(p.y * grid_dim),0),grid_dim - 1);
    cell_keys[idx] = gy * grid_dim + gx;
}

// Kernel 2 : Initialize the bounding super-triangle
__global__ void init_super_triangle(float2* points, int3* triangles, int3* neighbors, int n) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    points[n] = make_float2(-10.0f,-10.0f);
    points[n + 1] = make_float2(10.0f,-10.0f);
    points[n + 2] = make_float2(0.0f, 10.0f);

    triangles[0] = make_int3(n,n + 1,n + 2);
    neighbors[0] = make_int3(-1,-1,-1);
}

// Kernel 3 : Fixing each point to its enclosing triangle and finally deciding the point which will split the enclosing triangle
__global__ void locate_and_vote(const float2* points, const int3* triangles, const int3* neighbors, int* point_tri, const int* inserted, int* tri_winner, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n || inserted[idx]) return;

    float2 p = points[idx];
    int t = point_tri[idx];

    // Loop until we reach a time when the point falls inside the 3 edges 
    for (int step = 0; step < 10000; ++step) {
        int3 v = triangles[t];
        int3 nbr = neighbors[t];

        float o0 = orient2d(points[v.y], points[v.z], p); 
        float o1 = orient2d(points[v.z], points[v.x], p); 
        float o2 = orient2d(points[v.x], points[v.y], p);

        // Move across the most negative edge (outside current triangle)
        if (o0 < o1 && o0 < o2 && o0 < -1e-7f && nbr.x != -1) {
            t = nbr.x;
        } else if (o1 < o2 && o1 < -1e-7f && nbr.y != -1) {
            t = nbr.y;
        } else if (o2 < -1e-7f && nbr.z != -1) {
            t = nbr.z;
        } else {
            break; // Breaking condition : Point is inside the triangle 
        }
    }

    point_tri[idx] = t;
    atomicMin(&tri_winner[t], idx); // Winner point
}

// Kernel 4 : Split the triangle into 3 sub-triangles using the winner point 
__global__ void split_triangles(int3* triangles, int3* neighbors, const int* tri_winner, int* inserted, int* split_base, int* num_triangles, int prev_num_triangles, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= prev_num_triangles) return;

    int p = tri_winner[idx];
    if (p >= n) return; 

    inserted[p] = 1;
    int base = atomicAdd(num_triangles, 2); // Allocating 2 new triangle slots for the split
    split_base[idx] = base;

    int3 v = triangles[idx];
    int3 nbr = neighbors[idx];

    int c0 = idx;
    int c1 = base;
    int c2 = base + 1;

    // 3 new sub-triangles sharing interior point p
    triangles[c0] = make_int3(p, v.y, v.z);
    triangles[c1] = make_int3(v.z, v.x, p);
    triangles[c2] = make_int3(v.x, v.y, p);

    // Assign internal and external neighbors opposite to v0, v1, v2 respectively
    neighbors[c0] = make_int3(nbr.x, c1, c2);
    neighbors[c1] = make_int3(c2, c0, nbr.y);
    neighbors[c2] = make_int3(c0, c1, nbr.z);
}

// Kernel 5 (Part 2): Update neighbor pointers when an adjacent triangle was split
__global__ void repair_neighbors(const int3* triangles, int3* neighbors, const int* split_base, int current_num_triangles) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= current_num_triangles) return;

    int3 v = triangles[t];
    int3 nbr = neighbors[t];

    // Helper function to find which of the 3 split children owns shared edge (u, w)
    auto resolve_neighbor = [&](int n_idx, int u, int w) -> int {
        if (n_idx == -1) return -1;
        int base = split_base[n_idx];
        if (base == -1) return n_idx; // Neighbor was not split
        if (has_edge(triangles[n_idx], u, w))    return n_idx;
        if (has_edge(triangles[base], u, w))     return base;
        return base + 1;
    };

    nbr.x = resolve_neighbor(nbr.x, v.y, v.z);
    nbr.y = resolve_neighbor(nbr.y, v.z, v.x);
    nbr.z = resolve_neighbor(nbr.z, v.x, v.y);
    neighbors[t] = nbr;
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

    compute_grid_keys<<<blocks, threads>>>(thrust::raw_pointer_cast(d_points.data()), thrust::raw_pointer_cast(d_cell_keys.data()), N, grid_dim);

    thrust::sort_by_key(d_cell_keys.begin(),d_cell_keys.end(),d_points.begin());

    init_super_triangle<<<1, 1>>>(thrust::raw_pointer_cast(d_points.data()), thrust::raw_pointer_cast(d_triangles.data()), thrust::raw_pointer_cast(d_neighbors.data()), N);

    std::cout << "Initialized " << N << " spatially sorted points and allocated "<< max_triangles << " triangle slots on GPU\n";

    thrust::device_vector<int> d_point_tri(N, 0);          
    thrust::device_vector<int> d_inserted(N, 0);          
    thrust::device_vector<int> d_tri_winner(max_triangles);
    thrust::device_vector<int> d_split_base(max_triangles, -1);
    thrust::device_vector<int> d_num_triangles(1, 1);

    int rounds = 0;
    int current_triangles = 1;
    while (current_triangles < max_triangles) {
        int prev_triangles = current_triangles;
        int t_blocks = (prev_triangles + threads - 1) / threads;

        thrust::fill(d_tri_winner.begin(), d_tri_winner.begin() + prev_triangles, INT_MAX);
        thrust::fill(d_split_base.begin(), d_split_base.begin() + prev_triangles, -1);

        locate_and_vote<<<blocks, threads>>>(thrust::raw_pointer_cast(d_points.data()), thrust::raw_pointer_cast(d_triangles.data()), thrust::raw_pointer_cast(d_neighbors.data()), thrust::raw_pointer_cast(d_point_tri.data()), thrust::raw_pointer_cast(d_inserted.data()), thrust::raw_pointer_cast(d_tri_winner.data()), N);

        split_triangles<<<t_blocks, threads>>>(thrust::raw_pointer_cast(d_triangles.data()), thrust::raw_pointer_cast(d_neighbors.data()), thrust::raw_pointer_cast(d_tri_winner.data()), thrust::raw_pointer_cast(d_inserted.data()), thrust::raw_pointer_cast(d_split_base.data()), thrust::raw_pointer_cast(d_num_triangles.data()), prev_triangles, N);

        current_triangles = d_num_triangles[0];
        int all_blocks = (current_triangles + threads - 1) / threads;

        repair_neighbors<<<all_blocks, threads>>>(thrust::raw_pointer_cast(d_triangles.data()), thrust::raw_pointer_cast(d_neighbors.data()), thrust::raw_pointer_cast(d_split_base.data()), current_triangles);

        rounds++;
        if (current_triangles == prev_triangles) break; 
    }

    cudaDeviceSynchronize();
    std::cout << "Inserted all " << N << " points in "<< rounds << " parallel rounds -> " << current_triangles << " triangles\n";

    return 0;
}
