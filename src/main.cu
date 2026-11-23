#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/fill.h>
#include <iostream>
#include <fstream>
#include <random>
#include <cmath>
#include <climits>

// -----------------------------------------------------------------------------
// Geometric Predicates & Helpers
// -----------------------------------------------------------------------------

// 2D orientation test: > 0 if (a, b, c) is counter-clockwise (CCW)
__device__ inline float orient2d(float2 a, float2 b, float2 c) {
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

// Exact 3x3 determinant in-circle test: returns true if D lies inside circumcircle of CCW (A, B, C)
__device__ inline bool is_in_circle(float2 a, float2 b, float2 c, float2 d) {
    float adx = a.x - d.x, ady = a.y - d.y;
    float bdx = b.x - d.x, bdy = b.y - d.y;
    float cdx = c.x - d.x, cdy = c.y - d.y;

    float alift = adx * adx + ady * ady;
    float blift = bdx * bdx + bdy * bdy;
    float clift = cdx * cdx + cdy * cdy;

    float det = adx * (bdy * clift - cdy * blift)
              - ady * (bdx * clift - cdx * blift)
              + alift * (bdx * cdy - cdx * bdy);

    return det > 0.0f; // Strict Delaunay check without artificial epsilon padding
}

// Helper: Check if triangle `tri` contains both vertex indices `u` and `w`
__device__ inline bool has_edge(int3 tri, int u, int w) {
    bool has_u = (tri.x == u || tri.y == u || tri.z == u);
    bool has_w = (tri.x == w || tri.y == w || tri.z == w);
    return has_u && has_w;
}

// Helper: Get vertex of triangle `tri` opposite to shared edge (u, w)
__device__ inline int get_opposite_vertex(int3 tri, int u, int w) {
    if (tri.x != u && tri.x != w) return tri.x;
    if (tri.y != u && tri.y != w) return tri.y;
    return tri.z;
}

// Helper: Retrieve neighbor pointer opposite edge (u, w)
__device__ inline int get_edge_neighbor(int3 tri, int3 nbr, int u, int w) {
    if ((tri.y == u && tri.z == w) || (tri.y == w && tri.z == u)) return nbr.x;
    if ((tri.z == u && tri.x == w) || (tri.z == w && tri.x == u)) return nbr.y;
    if ((tri.x == u && tri.y == w) || (tri.x == w && tri.y == u)) return nbr.z;
    return -1;
}

// -----------------------------------------------------------------------------
// Part 1 & Part 2 Kernels: Point Location & 1-to-3 Triangle Splitting
// -----------------------------------------------------------------------------

__global__ void compute_grid_keys(const float2* points, int* cell_keys, int n, int grid_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float2 p = points[idx];
    int gx = min(max(int(p.x * grid_dim), 0), grid_dim - 1);
    int gy = min(max(int(p.y * grid_dim), 0), grid_dim - 1);
    cell_keys[idx] = gy * grid_dim + gx;
}

__global__ void init_super_triangle(float2* points, int3* triangles, int3* neighbors, int n) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // Tighter super-triangle bounds enclosing [0,1]x[0,1]
    points[n]     = make_float2(-1.5f, -0.5f);
    points[n + 1] = make_float2( 2.5f, -0.5f);
    points[n + 2] = make_float2( 0.5f,  2.5f);

    triangles[0] = make_int3(n, n + 1, n + 2);
    neighbors[0] = make_int3(-1, -1, -1);
}

__global__ void locate_and_vote(
    const float2* points,
    const int3* triangles,
    const int3* neighbors,
    int* point_tri,
    const int* inserted,
    int* tri_winner,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n || inserted[idx]) return;

    float2 p = points[idx];
    int t = point_tri[idx];

    for (int step = 0; step < 10000; ++step) {
        int3 v = triangles[t];
        int3 nbr = neighbors[t];

        float o0 = orient2d(points[v.y], points[v.z], p); 
        float o1 = orient2d(points[v.z], points[v.x], p); 
        float o2 = orient2d(points[v.x], points[v.y], p);

        int next_t = -1;
        float min_o = -1e-7f;

        if (o0 < min_o && nbr.x != -1) { min_o = o0; next_t = nbr.x; }
        if (o1 < min_o && nbr.y != -1) { min_o = o1; next_t = nbr.y; }
        if (o2 < min_o && nbr.z != -1) { min_o = o2; next_t = nbr.z; }

        if (next_t != -1) {
            t = next_t;
        } else {
            break;
        }
    }

    point_tri[idx] = t;
    atomicMin(&tri_winner[t], idx);
}

__global__ void split_triangles(
    int3* triangles,
    int3* neighbors,
    const int* tri_winner,
    int* inserted,
    int* split_base,
    int* num_triangles,
    int prev_num_triangles,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= prev_num_triangles) return;

    int p = tri_winner[idx];
    if (p >= n) return;

    inserted[p] = 1;
    int base = atomicAdd(num_triangles, 2);
    split_base[idx] = base;

    int3 v = triangles[idx];
    int3 nbr = neighbors[idx];

    int c0 = idx;
    int c1 = base;
    int c2 = base + 1;

    triangles[c0] = make_int3(p,   v.y, v.z);
    triangles[c1] = make_int3(v.z, v.x, p);
    triangles[c2] = make_int3(v.x, v.y, p);

    neighbors[c0] = make_int3(nbr.x, c1,    c2);
    neighbors[c1] = make_int3(c2,    c0,    nbr.y);
    neighbors[c2] = make_int3(c0,    c1,    nbr.z);
}

__global__ void repair_neighbors(
    const int3* triangles,
    int3* neighbors,
    const int* split_base,
    int current_num_triangles
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= current_num_triangles) return;

    int3 v = triangles[t];
    int3 nbr = neighbors[t];

    auto resolve_neighbor = [&](int n_idx, int u, int w) -> int {
        if (n_idx == -1) return -1;
        int base = split_base[n_idx];
        if (base == -1) return n_idx;
        if (has_edge(triangles[n_idx], u, w)) return n_idx;
        if (has_edge(triangles[base], u, w))  return base;
        return base + 1;
    };

    nbr.x = resolve_neighbor(nbr.x, v.y, v.z);
    nbr.y = resolve_neighbor(nbr.y, v.z, v.x);
    nbr.z = resolve_neighbor(nbr.z, v.x, v.y);
    neighbors[t] = nbr;
}

// -----------------------------------------------------------------------------
// Part 3 Kernels: Synchronized Parallel Lawson Edge Flipping
// -----------------------------------------------------------------------------

__global__ void legalize_edges(
    const float2* points,
    int3* triangles,
    int3* neighbors,
    int* lock,
    int* flip_count,
    int num_triangles
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= num_triangles) return;

    int3 v_t = triangles[t];
    int3 nbr_t = neighbors[t];

    int nbrs[3]   = {nbr_t.x, nbr_t.y, nbr_t.z};
    int2 edges[3] = {make_int2(v_t.y, v_t.z), make_int2(v_t.z, v_t.x), make_int2(v_t.x, v_t.y)};
    int opps[3]   = {v_t.x, v_t.y, v_t.z};

    for (int i = 0; i < 3; ++i) {
        int n = nbrs[i];
        if (n <= t) continue;

        int A = opps[i];
        int B = edges[i].x;
        int C = edges[i].y;

        int3 v_n = triangles[n];
        int3 nbr_n = neighbors[n];

        int D = get_opposite_vertex(v_n, B, C);

        int n_BD = get_edge_neighbor(v_n, nbr_n, B, D);
        int n_DC = get_edge_neighbor(v_n, nbr_n, D, C);
        int n_AB = get_edge_neighbor(v_t, nbr_t, A, B);
        int n_CA = get_edge_neighbor(v_t, nbr_t, C, A);

        if (is_in_circle(points[A], points[B], points[C], points[D])) {
            if (atomicCAS(&lock[t], 0, 1) == 0) {
                if (atomicCAS(&lock[n], 0, 1) == 0) {
                    triangles[t] = make_int3(A, B, D);
                    triangles[n] = make_int3(A, D, C);

                    neighbors[t] = make_int3(n_BD, n, n_AB);
                    neighbors[n] = make_int3(n_DC, n_CA, t);

                    atomicAdd(flip_count, 1);

                    atomicExch(&lock[n], 0);
                }
                atomicExch(&lock[t], 0);
            }
        }
    }
}

__global__ void repair_flipped_neighbors(
    const int3* triangles,
    int3* neighbors,
    int num_triangles
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= num_triangles) return;

    int3 v = triangles[t];
    int3 nbr = neighbors[t];

    auto fix_nbr = [&](int n_idx, int u, int w) -> int {
        if (n_idx == -1) return -1;
        if (has_edge(triangles[n_idx], u, w)) return n_idx;

        int3 n_nbr = neighbors[n_idx];
        int cand[3] = {n_nbr.x, n_nbr.y, n_nbr.z};
        for (int c = 0; c < 3; ++c) {
            if (cand[c] != -1 && has_edge(triangles[cand[c]], u, w)) {
                return cand[c];
            }
        }
        return n_idx;
    };

    nbr.x = fix_nbr(nbr.x, v.y, v.z);
    nbr.y = fix_nbr(nbr.y, v.z, v.x);
    nbr.z = fix_nbr(nbr.z, v.x, v.y);

    neighbors[t] = nbr;
}

// -----------------------------------------------------------------------------
// Main Host Driver
// -----------------------------------------------------------------------------

int main() {
    const int N = 30; // Point count
    const int max_triangles = 2 * N + 1;
    const int grid_dim = static_cast<int>(std::sqrt(N));

    thrust::host_vector<float2> h_points(N + 3);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.05f, 0.95f);
    for (int i = 0; i < N; ++i) h_points[i] = make_float2(dist(rng), dist(rng));

    thrust::device_vector<float2> d_points = h_points;
    thrust::device_vector<int> d_cell_keys(N);
    thrust::device_vector<int3> d_triangles(max_triangles);
    thrust::device_vector<int3> d_neighbors(max_triangles);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    compute_grid_keys<<<blocks, threads>>>(
        thrust::raw_pointer_cast(d_points.data()),
        thrust::raw_pointer_cast(d_cell_keys.data()),
        N, grid_dim
    );
    thrust::sort_by_key(d_cell_keys.begin(), d_cell_keys.end(), d_points.begin());

    init_super_triangle<<<1, 1>>>(
        thrust::raw_pointer_cast(d_points.data()),
        thrust::raw_pointer_cast(d_triangles.data()),
        thrust::raw_pointer_cast(d_neighbors.data()),
        N
    );

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

        locate_and_vote<<<blocks, threads>>>(
            thrust::raw_pointer_cast(d_points.data()),
            thrust::raw_pointer_cast(d_triangles.data()),
            thrust::raw_pointer_cast(d_neighbors.data()),
            thrust::raw_pointer_cast(d_point_tri.data()),
            thrust::raw_pointer_cast(d_inserted.data()),
            thrust::raw_pointer_cast(d_tri_winner.data()),
            N
        );

        split_triangles<<<t_blocks, threads>>>(
            thrust::raw_pointer_cast(d_triangles.data()),
            thrust::raw_pointer_cast(d_neighbors.data()),
            thrust::raw_pointer_cast(d_tri_winner.data()),
            thrust::raw_pointer_cast(d_inserted.data()),
            thrust::raw_pointer_cast(d_split_base.data()),
            thrust::raw_pointer_cast(d_num_triangles.data()),
            prev_triangles,
            N
        );

        current_triangles = d_num_triangles[0];
        int all_blocks = (current_triangles + threads - 1) / threads;

        repair_neighbors<<<all_blocks, threads>>>(
            thrust::raw_pointer_cast(d_triangles.data()),
            thrust::raw_pointer_cast(d_neighbors.data()),
            thrust::raw_pointer_cast(d_split_base.data()),
            current_triangles
        );

        rounds++;
        if (current_triangles == prev_triangles) break;
    }

    thrust::device_vector<int> d_lock(max_triangles, 0);
    thrust::device_vector<int> d_flip_count(1, 0);

    int flip_rounds = 0;
    int total_flips = 0;
    int t_all_blocks = (current_triangles + threads - 1) / threads;

    while (true) {
        d_flip_count[0] = 0;

        legalize_edges<<<t_all_blocks, threads>>>(
            thrust::raw_pointer_cast(d_points.data()),
            thrust::raw_pointer_cast(d_triangles.data()),
            thrust::raw_pointer_cast(d_neighbors.data()),
            thrust::raw_pointer_cast(d_lock.data()),
            thrust::raw_pointer_cast(d_flip_count.data()),
            current_triangles
        );

        repair_flipped_neighbors<<<t_all_blocks, threads>>>(
            thrust::raw_pointer_cast(d_triangles.data()),
            thrust::raw_pointer_cast(d_neighbors.data()),
            current_triangles
        );

        int flips_this_round = d_flip_count[0];
        total_flips += flips_this_round;
        flip_rounds++;

        if (flips_this_round == 0 || flip_rounds > 1000) break;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "====================================================\n";
    std::cout << " GPU Delaunay Triangulation Complete!\n";
    std::cout << "====================================================\n";
    std::cout << " Points Inserted   : " << N << "\n";
    std::cout << " Final Triangles   : " << current_triangles << "\n";
    std::cout << " GPU Execution Time: " << milliseconds << " ms\n";
    std::cout << "====================================================\n";

    h_points = d_points;

    std::ofstream p_file("points.txt");
    for (int i = 0; i < N; ++i) {
        p_file << h_points[i].x << " " << h_points[i].y << "\n";
    }
    p_file.close();

    thrust::host_vector<int3> h_triangles = d_triangles;
    std::ofstream t_file("triangles.txt");
    for (int i = 0; i < current_triangles; ++i) {
        int3 t = h_triangles[i];
        if (t.x < N && t.y < N && t.z < N) {
            t_file << t.x << " " << t.y << " " << t.z << "\n";
        }
    }
    t_file.close();

    return 0;
}