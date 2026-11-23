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
#include <algorithm>

// -----------------------------------------------------------------------------
// Geometric Predicates & Helpers
// -----------------------------------------------------------------------------

// 2D orientation test in double precision: > 0 if (a, b, c) is counter-clockwise (CCW)
__device__ inline double orient2d(float2 a, float2 b, float2 c) {
    return (double(b.x) - double(a.x)) * (double(c.y) - double(a.y))
         - (double(b.y) - double(a.y)) * (double(c.x) - double(a.x));
}

// 3x3 determinant in-circle test in double precision
__device__ inline bool is_in_circle(float2 a, float2 b, float2 c, float2 d) {
    double adx = double(a.x) - double(d.x), ady = double(a.y) - double(d.y);
    double bdx = double(b.x) - double(d.x), bdy = double(b.y) - double(d.y);
    double cdx = double(c.x) - double(d.x), cdy = double(c.y) - double(d.y);

    double alift = adx * adx + ady * ady;
    double blift = bdx * bdx + bdy * bdy;
    double clift = cdx * cdx + cdy * cdy;

    double det = adx * (bdy * clift - cdy * blift)
               - ady * (bdx * clift - cdx * blift)
               + alift * (bdx * cdy - cdx * bdy);

    return det > 1e-12;
}

__device__ inline bool has_edge(int3 tri, int u, int w) {
    bool has_u = (tri.x == u || tri.y == u || tri.z == u);
    bool has_w = (tri.x == w || tri.y == w || tri.z == w);
    return has_u && has_w;
}

__device__ inline int get_opposite_vertex(int3 tri, int u, int w) {
    if (tri.x != u && tri.x != w) return tri.x;
    if (tri.y != u && tri.y != w) return tri.y;
    return tri.z;
}

__device__ inline int get_edge_neighbor(int3 tri, int3 nbr, int u, int w) {
    if ((tri.y == u && tri.z == w) || (tri.y == w && tri.z == u)) return nbr.x;
    if ((tri.z == u && tri.x == w) || (tri.z == w && tri.x == u)) return nbr.y;
    if ((tri.x == u && tri.y == w) || (tri.x == w && tri.y == u)) return nbr.z;
    return -1;
}

__device__ inline int resolve_split_neighbor(
    const int3* triangles,
    const int* split_base,
    int n_idx,
    int u,
    int w
) {
    if (n_idx == -1) return -1;
    int base = split_base[n_idx];
    if (base == -1) return n_idx;
    if (has_edge(triangles[n_idx], u, w)) return n_idx;
    if (has_edge(triangles[base], u, w))  return base;
    return base + 1;
}

__device__ inline int resolve_flipped_neighbor(
    const int3* triangles,
    const int* flip_partner,
    int n_idx,
    int u,
    int w
) {
    if (n_idx == -1) return -1;
    if (has_edge(triangles[n_idx], u, w)) return n_idx;
    int partner = flip_partner[n_idx];
    if (partner != -1 && has_edge(triangles[partner], u, w)) return partner;
    return n_idx;
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

    // Large super-triangle so its vertices stay outside circumcircles of hull triplets
    points[n]     = make_float2(-100.0f, -100.0f);
    points[n + 1] = make_float2( 100.0f, -100.0f);
    points[n + 2] = make_float2(   0.0f,  100.0f);

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

        double o0 = orient2d(points[v.y], points[v.z], p);
        double o1 = orient2d(points[v.z], points[v.x], p);
        double o2 = orient2d(points[v.x], points[v.y], p);

        int next_t = -1;
        double min_o = -1e-12;

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

    nbr.x = resolve_split_neighbor(triangles, split_base, nbr.x, v.y, v.z);
    nbr.y = resolve_split_neighbor(triangles, split_base, nbr.y, v.z, v.x);
    nbr.z = resolve_split_neighbor(triangles, split_base, nbr.z, v.x, v.y);
    neighbors[t] = nbr;
}

// -----------------------------------------------------------------------------
// Part 3 Kernels: Conflict-Free 2-Phase Parallel Lawson Edge Flipping
// -----------------------------------------------------------------------------

__global__ void propose_flips(
    const float2* points,
    const int3* triangles,
    const int3* neighbors,
    int* tri_lock,
    int* flip_edge,
    int* flip_partner,
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
        int D = get_opposite_vertex(v_n, B, C);

        // Verify empty circumcircle and strict convexity of quadrilateral A-B-D-C
        if (is_in_circle(points[A], points[B], points[C], points[D]) &&
            orient2d(points[A], points[B], points[D]) > 0.0 &&
            orient2d(points[A], points[D], points[C]) > 0.0) {
            if (atomicCAS(&tri_lock[t], -1, t) == -1) {
                if (atomicCAS(&tri_lock[n], -1, t) == -1) {
                    flip_edge[t] = i;
                    flip_partner[t] = n;
                    flip_partner[n] = t;
                    atomicAdd(flip_count, 1);
                    return; // Participate in at most one flip per round
                } else {
                    atomicExch(&tri_lock[t], -1);
                }
            }
        }
    }
}

__global__ void execute_flips(
    int3* triangles,
    int3* neighbors,
    const int* flip_edge,
    const int* flip_partner,
    int num_triangles
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= num_triangles) return;

    int i = flip_edge[t];
    if (i == -1) return;

    int n = flip_partner[t];

    int3 v_t = triangles[t];
    int3 nbr_t = neighbors[t];
    int3 v_n = triangles[n];
    int3 nbr_n = neighbors[n];

    int A, B, C;
    if (i == 0)      { A = v_t.x; B = v_t.y; C = v_t.z; }
    else if (i == 1) { A = v_t.y; B = v_t.z; C = v_t.x; }
    else             { A = v_t.z; B = v_t.x; C = v_t.y; }

    int D = get_opposite_vertex(v_n, B, C);

    int n_BD = get_edge_neighbor(v_n, nbr_n, B, D);
    int n_DC = get_edge_neighbor(v_n, nbr_n, D, C);
    int n_AB = get_edge_neighbor(v_t, nbr_t, A, B);
    int n_CA = get_edge_neighbor(v_t, nbr_t, C, A);

    triangles[t] = make_int3(A, B, D);
    triangles[n] = make_int3(A, D, C);

    neighbors[t] = make_int3(n_BD, n, n_AB);
    neighbors[n] = make_int3(n_DC, n_CA, t);
}

__global__ void repair_flipped_neighbors(
    const int3* triangles,
    int3* neighbors,
    const int* flip_partner,
    int num_triangles
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= num_triangles) return;

    int3 v = triangles[t];
    int3 nbr = neighbors[t];

    nbr.x = resolve_flipped_neighbor(triangles, flip_partner, nbr.x, v.y, v.z);
    nbr.y = resolve_flipped_neighbor(triangles, flip_partner, nbr.y, v.z, v.x);
    nbr.z = resolve_flipped_neighbor(triangles, flip_partner, nbr.z, v.x, v.y);

    neighbors[t] = nbr;
}

// -----------------------------------------------------------------------------
// Main Host Driver
// -----------------------------------------------------------------------------

int main() {
    const int N = 10000;
    const int max_triangles = 2 * N + 1;
    const int grid_dim = std::max(1, static_cast<int>(std::sqrt(N)));

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

        if (current_triangles == prev_triangles) break;
    }

    thrust::device_vector<int> d_tri_lock(current_triangles, -1);
    thrust::device_vector<int> d_flip_edge(current_triangles, -1);
    thrust::device_vector<int> d_flip_partner(current_triangles, -1);
    thrust::device_vector<int> d_flip_count(1, 0);

    int flip_rounds = 0;
    int total_flips = 0;
    int t_all_blocks = (current_triangles + threads - 1) / threads;

    while (true) {
        thrust::fill(d_tri_lock.begin(), d_tri_lock.end(), -1);
        thrust::fill(d_flip_edge.begin(), d_flip_edge.end(), -1);
        thrust::fill(d_flip_partner.begin(), d_flip_partner.end(), -1);
        d_flip_count[0] = 0;

        propose_flips<<<t_all_blocks, threads>>>(
            thrust::raw_pointer_cast(d_points.data()),
            thrust::raw_pointer_cast(d_triangles.data()),
            thrust::raw_pointer_cast(d_neighbors.data()),
            thrust::raw_pointer_cast(d_tri_lock.data()),
            thrust::raw_pointer_cast(d_flip_edge.data()),
            thrust::raw_pointer_cast(d_flip_partner.data()),
            thrust::raw_pointer_cast(d_flip_count.data()),
            current_triangles
        );

        int flips_this_round = d_flip_count[0];
        if (flips_this_round == 0 || flip_rounds > 1000) break;

        execute_flips<<<t_all_blocks, threads>>>(
            thrust::raw_pointer_cast(d_triangles.data()),
            thrust::raw_pointer_cast(d_neighbors.data()),
            thrust::raw_pointer_cast(d_flip_edge.data()),
            thrust::raw_pointer_cast(d_flip_partner.data()),
            current_triangles
        );

        repair_flipped_neighbors<<<t_all_blocks, threads>>>(
            thrust::raw_pointer_cast(d_triangles.data()),
            thrust::raw_pointer_cast(d_neighbors.data()),
            thrust::raw_pointer_cast(d_flip_partner.data()),
            current_triangles
        );

        total_flips += flips_this_round;
        flip_rounds++;
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
    std::cout << " Total Edge Flips  : " << total_flips << " (" << flip_rounds << " rounds)\n";
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