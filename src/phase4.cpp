/**
 * APMA 2822B Final Project
 * Phase 4: Distributed MPI + GPU Conjugate Gradient Solver (HIP + MPI)
 *
 * Solves -Δu = f on the unit square with homogeneous Dirichlet boundary
 * conditions using a 5-point finite-difference stencil. The domain is
 * decomposed into horizontal slabs across MPI ranks, and each rank offloads
 * its local work to a GPU using HIP. MPI is assumed to be GPU-aware so that
 * halo exchanges can operate directly on device memory.
 *
 * Build modes:
 *   SingleMode  : single sine eigenmode with known exact solution
 *   MultiMode   : linear combination of several eigenmodes
 *   GenericRHS  : Gaussian bump RHS; no exact solution available
 *
 * Usage:
 *   mpirun -np P ./phase4 [Nx] [tol] [max_iters] [BuildMode] [csv_path]
 *
 *   Nx        : grid size in each dimension (Nx x Nx grid, default 1024)
 *   tol       : relative residual tolerance (default 1e-8)
 *   max_iters : maximum CG iterations (default 10000)
 *   BuildMode : 0 = SingleMode, 1 = MultiMode, 2 = GenericRHS
 *   csv_path  : output CSV path (default "results_phase4.csv")
 */

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <mpi.h>
#include <hip/hip_runtime.h>

// ============================================================================
// HIP error checking
// ============================================================================

#define HIP_CHECK(command)                                                    \
    do {                                                                      \
        hipError_t err = (command);                                           \
        if (err != hipSuccess) {                                              \
            std::cerr << "HIP error: " << hipGetErrorString(err)              \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;  \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

// ============================================================================
// Problem configuration and helpers
// ============================================================================

enum BuildMode {
    SingleMode,   // single sine eigenmode with known exact solution
    MultiMode,    // combination of several eigenmodes
    GenericRHS    // generic RHS (no exact solution)
};

std::string modeToString(BuildMode m) {
    switch (m) {
        case SingleMode:  return "SingleMode";
        case MultiMode:   return "MultiMode";
        case GenericRHS:  return "GenericRHS";
        default:          return "UNKNOWN";
    }
}

const double PI = 4.0 * std::atan(1.0);

// 1D index helper. Marked __host__ __device__ because kernels use it.
__host__ __device__
inline int idx(int i, int j, int Ny) {
    return i * Ny + j;
}

// True if (i, j) lies on the physical boundary of the global Nx x Ny grid
inline bool is_boundary(int i, int j, int Nx, int Ny) {
    return (i == 0 || i == Nx - 1 || j == 0 || j == Ny - 1);
}

// ============================================================================
// RHS and exact solution assembly
// ============================================================================

// Assemble RHS f and exact solution u (if known) on this rank’s slab.
//
// Local rows: 0           ... owned+1
//             ^              ^
//           top halo     bottom halo
//
// Interior rows owned by this rank are 1..owned, corresponding to global
// rows [global_start .. global_start+owned-1].
void build(int Nx, int Ny,
           int global_start,
           int owned,
           double h,
           std::vector<double>& rhs,
           std::vector<double>& exact,
           BuildMode mode)
{
    for (int il = 1; il <= owned; ++il) {
        const int i_global = global_start + (il - 1);
        const double x     = i_global * h;

        for (int j = 0; j < Ny; ++j) {
            const int index = idx(il, j, Ny);

            // Physical boundary: u = 0, f = 0
            if (is_boundary(i_global, j, Nx, Ny)) {
                exact[index] = 0.0;
                rhs[index]   = 0.0;
                continue;
            }

            // Interior point
            const double y  = j * h;
            const double u1 = std::sin(PI * x) * std::sin(PI * y);  // (1,1), λ1 = 2π²
            const double f1 = (2.0 * PI * PI) * u1;                 // -Δ u1 = 2π² u1

            switch (mode) {
                case SingleMode: {
                    // Single eigenmode
                    exact[index] = u1;
                    rhs[index]   = f1;
                    break;
                }
                case MultiMode: {
                    // u = u1 + 0.3 u2 + 0.2 u3
                    const double u2 = std::sin(2.0 * PI * x) * std::sin(3.0 * PI * y); // (2,3), λ2 = 13π²
                    const double u3 = std::sin(4.0 * PI * x) * std::sin(2.0 * PI * y); // (4,2), λ3 = 20π²
                    const double u  = u1 + 0.3 * u2 + 0.2 * u3;
                    const double f  = f1
                                    + 13.0 * PI * PI * 0.3 * u2
                                    + 20.0 * PI * PI * 0.2 * u3;
                    exact[index] = u;
                    rhs[index]   = f;
                    break;
                }
                case GenericRHS: {
                    // Gaussian bump centered at (0.5, 0.5); no exact solution
                    const double x0    = 0.5;
                    const double y0    = 0.5;
                    const double sigma = 0.1;
                    const double dx    = x - x0;
                    const double dy    = y - y0;
                    const double r2    = dx * dx + dy * dy;
                    rhs[index]   = std::exp(-r2 / (2.0 * sigma * sigma));
                    exact[index] = 0.0;  // unknown solution
                    break;
                }
            }
        }
    }
}

// ============================================================================
// Error metrics (CPU-side)
// ============================================================================

// Compute global L2 and max error ||u - exact|| over all ranks
void compute_error_parallel(const std::vector<double>& u,
                            const std::vector<double>& exact,
                            int owned,
                            int Ny,
                            double& l2_err,
                            double& max_err)
{
    double local_sum_sq = 0.0;
    double local_max    = 0.0;

    // Local interior rows (1..owned), all columns
    for (int i = 1; i <= owned; ++i) {
        for (int j = 0; j < Ny; ++j) {
            const int    k    = idx(i, j, Ny);
            const double diff = std::abs(u[k] - exact[k]);
            local_sum_sq += diff * diff;
            local_max     = std::max(local_max, diff);
        }
    }

    double global_sum_sq = 0.0;
    double global_max    = 0.0;

    MPI_Allreduce(&local_sum_sq, &global_sum_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_max,    &global_max,    1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    l2_err  = std::sqrt(global_sum_sq);
    max_err = global_max;
}

// ============================================================================
// MPI halo exchange (GPU-aware)
// ============================================================================

// Exchange top/bottom halo rows of p with neighbors. This assumes
// GPU-aware MPI so that device pointers are valid in MPI calls.
inline void exchange_halos(double* d_p,
                           int Ny,
                           int owned,
                           int top,
                           int bottom)
{
    double* top_halo    = d_p;                     // row 0
    double* first_owned = d_p + Ny;                // row 1
    double* last_owned  = d_p + owned * Ny;        // row owned
    double* bottom_halo = d_p + (owned + 1) * Ny;  // row owned+1

    MPI_Request reqs[4];
    int r = 0;

    MPI_Irecv(top_halo,    Ny, MPI_DOUBLE, top,    0, MPI_COMM_WORLD, &reqs[r++]);
    MPI_Irecv(bottom_halo, Ny, MPI_DOUBLE, bottom, 1, MPI_COMM_WORLD, &reqs[r++]);

    MPI_Isend(first_owned, Ny, MPI_DOUBLE, top,    1, MPI_COMM_WORLD, &reqs[r++]);
    MPI_Isend(last_owned,  Ny, MPI_DOUBLE, bottom, 0, MPI_COMM_WORLD, &reqs[r++]);

    MPI_Waitall(r, reqs, MPI_STATUSES_IGNORE);

    // For physical boundaries, enforce Dirichlet u=0 in halo rows
    if (top == MPI_PROC_NULL) {
        HIP_CHECK(hipMemset(top_halo, 0, Ny * sizeof(double)));
    }
    if (bottom == MPI_PROC_NULL) {
        HIP_CHECK(hipMemset(bottom_halo, 0, Ny * sizeof(double)));
    }
}

// ============================================================================
// CG statistics and timing structs
// ============================================================================

struct CGResult {
    int    iters      = 0;
    double norm0      = 0.0;
    double norm_final = 0.0;
    bool   converged  = false;
};

struct DeviceBuffers {
    double* d_u   = nullptr;  // solution
    double* d_r   = nullptr;  // residual
    double* d_p   = nullptr;  // search direction
    double* d_Ap  = nullptr;  // A p
    double* d_sum = nullptr;  // workspace for dot products
    int     n     = 0;

    void allocate(int n_local) {
        n = n_local;
        HIP_CHECK(hipMalloc(&d_u,   n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_r,   n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_p,   n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_Ap,  n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_sum, n * sizeof(double)));
    }

    void free() {
        HIP_CHECK(hipFree(d_u));
        HIP_CHECK(hipFree(d_r));
        HIP_CHECK(hipFree(d_p));
        HIP_CHECK(hipFree(d_Ap));
        HIP_CHECK(hipFree(d_sum));
    }

    ~DeviceBuffers() { free(); }
};

struct TimingData {
    double t_halo     = 0.0;  // halo exchange
    double t_applyA   = 0.0;  // Ap = A p
    double t_axpy_u   = 0.0;  // u = u + alpha p
    double t_axpy_r   = 0.0;  // r = r - alpha Ap
    double t_update_p = 0.0;  // p = r + beta p
    double t_dot      = 0.0;  // total dot time in loop
    double t_dot_init = 0.0;  // initial (r,r)
    double t_dot_pAp  = 0.0;  // (p,Ap)
    double t_dot_rr   = 0.0;  // (r,r) inside loop
    double t_total    = 0.0;  // full CG loop time (sum over iterations)
};

struct DotTiming {
    double t_gpu       = 0.0; // time spent in GPU kernels + sync
    double t_memcpy    = 0.0; // device→host copy of scalar
    double t_allreduce = 0.0; // MPI_Allreduce time
};

// ============================================================================
// GPU kernels
// ============================================================================

// Apply 5-point Laplacian: y = A x, on local interior rows 1..owned.
// Column boundaries j=0 and j=Ny-1 are Dirichlet (y=0).
__global__
void apply_A_kernel(const double* __restrict__ x,
                    double*       __restrict__ y,
                    int owned,
                    int Ny,
                    double inv_h2)
{
    const int i = blockIdx.y * blockDim.y + threadIdx.y + 1; // interior rows: 1..owned
    const int j = blockIdx.x * blockDim.x + threadIdx.x;     // columns: 0..Ny-1

    if (i < 1 || i > owned || j < 0 || j >= Ny) return;

    const int index = idx(i, j, Ny);

    // Left/right physical boundary columns
    if (j == 0 || j == Ny - 1) {
        y[index] = 0.0;
        return;
    }

    const double center = x[index];
    const double up     = x[idx(i - 1, j,     Ny)];
    const double down   = x[idx(i + 1, j,     Ny)];
    const double left   = x[idx(i,     j - 1, Ny)];
    const double right  = x[idx(i,     j + 1, Ny)];

    y[index] = (4.0 * center - up - down - left - right) * inv_h2;
}

// Thin wrapper that configures the grid and launches apply_A_kernel
inline void apply_A(double* d_x,
                    double* d_y,
                    int owned,
                    int Ny,
                    double inv_h2)
{
    dim3 block(32, 8);
    dim3 grid((Ny    + block.x - 1) / block.x,
              (owned + block.y - 1) / block.y);

    apply_A_kernel<<<grid, block>>>(d_x, d_y, owned, Ny, inv_h2);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
}

// y <- y + alpha * x
__global__
void axpy_kernel(int n,
                 double alpha,
                 const double* __restrict__ x,
                 double*       __restrict__ y)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    y[k] += alpha * x[k];
}

// p <- r + beta * p
__global__
void update_p_kernel(int n,
                     double beta,
                     const double* __restrict__ r,
                     double*       __restrict__ p)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    p[k] = r[k] + beta * p[k];
}

// Dot product kernels: two-stage reduction

// Stage 1: partial sums over local_size elements into partial[blockIdx.x]
__global__
void dot_partial_kernel(const double* __restrict__ a,
                        const double* __restrict__ b,
                        double*       __restrict__ partial,
                        int n)
{
    extern __shared__ double sdata[];

    const int tid = threadIdx.x;
    const int i   = blockIdx.x * blockDim.x + threadIdx.x;

    double val = 0.0;
    if (i < n) {
        val = a[i] * b[i];
    }

    sdata[tid] = val;
    __syncthreads();

    // Block-level reduction to sdata[0]
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial[blockIdx.x] = sdata[0];
    }
}

// Stage 2: reduce partial[0..num_blocks-1] into result[0]
__global__
void dot_reduce_kernel(const double* __restrict__ partial,
                       double*       __restrict__ result,
                       int num_blocks)
{
    extern __shared__ double sdata[];

    const int tid = threadIdx.x;
    double sum    = 0.0;

    // Each thread accumulates a strided subset of partial[]
    for (int i = tid; i < num_blocks; i += blockDim.x) {
        sum += partial[i];
    }
    sdata[tid] = sum;
    __syncthreads();

    // Reduction within the block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *result = sdata[0];
    }
}

// ============================================================================
// Dot product wrapper (GPU + MPI_Allreduce)
// ============================================================================

double dot(const double* d_a,
           const double* d_b,
           double*       d_sum,
           int           local_size,
           DotTiming&    dt)
{
    const int threads = 256;
    const int blocks  = (local_size + threads - 1) / threads;

    const std::size_t shmem = threads * sizeof(double);

    // Use d_sum[0..blocks-1] as partials, then d_sum[0] as final result.

    // GPU kernels + device sync
    const double t0 = MPI_Wtime();

    // Stage 1: per-block partial sums
    dot_partial_kernel<<<blocks, threads, shmem>>>(d_a, d_b, d_sum, local_size);
    HIP_CHECK(hipGetLastError());

    // Stage 2: single-block reduction of partials
    const int red_threads = 256;
    const std::size_t red_shmem = red_threads * sizeof(double);
    dot_reduce_kernel<<<1, red_threads, red_shmem>>>(d_sum, d_sum, blocks);
    HIP_CHECK(hipGetLastError());

    HIP_CHECK(hipDeviceSynchronize());
    const double t1 = MPI_Wtime();
    dt.t_gpu += (t1 - t0);

    // Copy scalar result to host
    double local_sum = 0.0;
    HIP_CHECK(hipMemcpy(&local_sum, d_sum, sizeof(double), hipMemcpyDeviceToHost));
    const double t2 = MPI_Wtime();
    dt.t_memcpy += (t2 - t1);

    // Global reduction across ranks
    double global_sum = 0.0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    const double t3 = MPI_Wtime();
    dt.t_allreduce += (t3 - t2);

    return global_sum;
}

// ============================================================================
// Conjugate Gradient (MPI + GPU)
// ============================================================================

void cg_solve(int Nx,
              int Ny,
              int local_size,   // local_rows * Ny
              int local_rows,   // rows including halos
              int owned,        // interior rows owned by this rank
              int top,
              int bottom,
              int rank,
              double h,
              std::vector<double>& u,  // host solution (includes halos)
              int max_iters,
              double tol,
              CGResult& result,
              DeviceBuffers& dev,
              TimingData& timer,
              DotTiming& dot_times)
{
    (void)Nx;            // Nx is implicit in owned/local_size
    (void)local_rows;    // kept for symmetry / future use

    const double h2     = h * h;
    const double inv_h2 = 1.0 / h2;

    // Initial residual: r = b - A u, but u was zero and dev.d_r already has b.
    double t0_dot = MPI_Wtime();
    double rho    = dot(dev.d_r, dev.d_r, dev.d_sum, local_size, dot_times);
    double t1_dot = MPI_Wtime();

    timer.t_dot_init = t1_dot - t0_dot;
    timer.t_dot     += timer.t_dot_init;

    const double norm0 = std::sqrt(rho);
    result.norm0       = norm0;

    if (norm0 < tol) {
        // rhs is effectively zero; nothing to do
        result.iters      = 0;
        result.norm_final = norm0;
        result.converged  = true;

        HIP_CHECK(hipMemcpy(u.data(), dev.d_u,
                            local_size * sizeof(double),
                            hipMemcpyDeviceToHost));
        return;
    }

    const int threads = 256;
    const int blocks  = (local_size + threads - 1) / threads;

    // Start CG iteration timer (per-rank, will be max-reduced later)
    const double t_cg_start = MPI_Wtime();

    for (int k = 0; k < max_iters; ++k) {

        // 1. Halo exchange for p
        double t0_halo = MPI_Wtime();
        exchange_halos(dev.d_p, Ny, owned, top, bottom);
        double t1_halo = MPI_Wtime();
        timer.t_halo  += (t1_halo - t0_halo);

        // 2. Ap = A p
        double t0_applyA = MPI_Wtime();
        apply_A(dev.d_p, dev.d_Ap, owned, Ny, inv_h2);
        double t1_applyA = MPI_Wtime();
        timer.t_applyA  += (t1_applyA - t0_applyA);

        // 3. alpha = (r,r) / (p,Ap)
        t0_dot = MPI_Wtime();
        double p_dot_Ap = dot(dev.d_p, dev.d_Ap, dev.d_sum, local_size, dot_times);
        t1_dot = MPI_Wtime();

        const double dt_pAp = t1_dot - t0_dot;
        timer.t_dot        += dt_pAp;
        timer.t_dot_pAp    += dt_pAp;

        if (p_dot_Ap <= 0.0) {
            if (rank == 0) {
                std::cerr << "CG breakdown: p^T A p <= 0 at iter " << k << "\n";
            }
            break;
        }

        const double alpha = rho / p_dot_Ap;

        // 4. u = u + alpha * p
        double t0_axpy_u = MPI_Wtime();
        axpy_kernel<<<blocks, threads>>>(local_size, alpha, dev.d_p, dev.d_u);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        double t1_axpy_u = MPI_Wtime();
        timer.t_axpy_u  += (t1_axpy_u - t0_axpy_u);

        // 5. r = r - alpha * Ap
        double t0_axpy_r = MPI_Wtime();
        axpy_kernel<<<blocks, threads>>>(local_size, -alpha, dev.d_Ap, dev.d_r);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        double t1_axpy_r = MPI_Wtime();
        timer.t_axpy_r  += (t1_axpy_r - t0_axpy_r);

        // 6. new_rho = (r,r), check convergence
        t0_dot = MPI_Wtime();
        double new_rho = dot(dev.d_r, dev.d_r, dev.d_sum, local_size, dot_times);
        t1_dot = MPI_Wtime();

        const double dt_rr = t1_dot - t0_dot;
        timer.t_dot       += dt_rr;
        timer.t_dot_rr    += dt_rr;

        const double norm = std::sqrt(new_rho);
        const double rel  = norm / norm0;

        if (rel < tol) {
            if (rank == 0) {
                std::cout << "converged in " << (k + 1)
                          << " iterations, rel = " << rel << "\n";
            }
            result.iters      = k + 1;
            result.norm_final = norm;
            result.converged  = true;

            timer.t_total += MPI_Wtime() - t_cg_start;

            HIP_CHECK(hipMemcpy(u.data(), dev.d_u,
                                local_size * sizeof(double),
                                hipMemcpyDeviceToHost));
            return;
        }

        // 7. p = r + beta * p
        const double beta = new_rho / rho;
        double t0_update_p = MPI_Wtime();
        update_p_kernel<<<blocks, threads>>>(local_size, beta, dev.d_r, dev.d_p);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        double t1_update_p = MPI_Wtime();
        timer.t_update_p  += (t1_update_p - t0_update_p);

        rho               = new_rho;
        result.iters      = k + 1;
        result.norm_final = norm;
    }

    // Exited loop without meeting tolerance
    timer.t_total += MPI_Wtime() - t_cg_start;
    result.converged = false;
}

// ============================================================================
// main
// ============================================================================

int main(int argc, char* argv[])
{
    // Defaults
    int        Nx        = 1024;
    int        Ny        = 1024;
    double     tolerance = 1e-8;
    int        max_iters = 10000;
    BuildMode  mode      = GenericRHS;
    std::string csv_path = "results_phase4.csv";

    if (argc > 1) { Nx = Ny = std::stoi(argv[1]); }
    if (argc > 2) { tolerance = std::stod(argv[2]); }
    if (argc > 3) { max_iters = std::stoi(argv[3]); }
    if (argc > 4) {
        const int m = std::stoi(argv[4]);
        if      (m == 0) mode = SingleMode;
        else if (m == 1) mode = MultiMode;
        else             mode = GenericRHS;
    }
    if (argc > 5) { csv_path = std::string(argv[5]); }

    MPI_Init(&argc, &argv);

    int rank   = -1;
    int nprocs = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (rank == 0) {
        std::cout << "Nx="        << Nx
                  << " tol="      << tolerance
                  << " max_iters="<< max_iters
                  << " mode="     << modeToString(mode)
                  << " csv=\""    << csv_path << "\"\n";
    }

    // One GPU per rank (round-robin if more ranks than devices)
    int device_count = 0;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    const int device_id = (device_count > 0) ? (rank % device_count) : 0;
    HIP_CHECK(hipSetDevice(device_id));

    // 1D slab decomposition in x-direction
    const int N_int      = Nx - 2; // global interior rows
    const int base       = N_int / nprocs;
    const int rem        = N_int % nprocs;

    const int owned      = base + (rank < rem ? 1 : 0); // interior rows on this rank
    const int start_off  = rank * base + std::min(rank, rem);

    const int local_rows = owned + 2;     // interior + 2 halo rows
    const int local_size = local_rows * Ny;

    const int top        = (rank == 0)          ? MPI_PROC_NULL : rank - 1;
    const int bottom     = (rank == nprocs - 1) ? MPI_PROC_NULL : rank + 1;

    const int global_start = 1 + start_off;     // first global interior row index

    const double h = 1.0 / (Nx - 1);

    // Host-side vectors (include halos)
    std::vector<double> u(local_size,    0.0);
    std::vector<double> rhs(local_size,  0.0);
    std::vector<double> exact(local_size, 0.0);

    // Assemble RHS and exact solution
    build(Nx, Ny, global_start, owned, h, rhs, exact, mode);

    // Device buffers
    DeviceBuffers dev;
    dev.allocate(local_size);

    // Copy initial data to device:
    //   u = 0, r = b, p = r
    HIP_CHECK(hipMemset(dev.d_u, 0, local_size * sizeof(double)));
    HIP_CHECK(hipMemcpy(dev.d_r, rhs.data(),
                        local_size * sizeof(double),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dev.d_p, dev.d_r,
                        local_size * sizeof(double),
                        hipMemcpyDeviceToDevice));

    // Run CG solve and time it (global max over ranks)
    CGResult  cg_res{};
    TimingData timer{};
    DotTiming  dot_times{};  // zero-initialize

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();
    cg_solve(Nx, Ny, local_size, local_rows, owned,
             top, bottom, rank, h,
             u, max_iters, tolerance,
             cg_res, dev, timer, dot_times);
    const double t1 = MPI_Wtime();

    const double local_time  = t1 - t0;
    double       runtime_sec = 0.0;
    MPI_Reduce(&local_time, &runtime_sec, 1,
               MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Collect per-phase CG timing (max over ranks)
    double local_t[10] = {
        timer.t_halo,      // 0
        timer.t_applyA,    // 1
        timer.t_axpy_u,    // 2
        timer.t_axpy_r,    // 3
        timer.t_update_p,  // 4
        timer.t_dot,       // 5
        timer.t_dot_init,  // 6
        timer.t_dot_pAp,   // 7
        timer.t_dot_rr,    // 8
        timer.t_total      // 9
    };

    double global_t[10] = {0.0};
    MPI_Reduce(local_t, global_t, 10, MPI_DOUBLE, MPI_MAX,
               0, MPI_COMM_WORLD);

    // Compute error metrics (only meaningful when exact solution is known)
    double l2_err  = -1.0;
    double max_err = -1.0;

    if (mode == GenericRHS) {
        if (rank == 0) {
            std::cout << "GenericRHS mode: skipping exact error check.\n";
        }
    } else {
        compute_error_parallel(u, exact, owned, Ny, l2_err, max_err);
        if (rank == 0) {
            std::cout << "L2 error   = " << l2_err  << "\n"
                      << "Max error  = " << max_err << "\n";
        }
    }

    // Final relative residual
    double rel_final = 0.0;
    if (cg_res.norm0 > 0.0) {
        rel_final = cg_res.norm_final / cg_res.norm0;
    }

    // Rank 0: append row to CSV with runtime breakdown
    if (rank == 0) {
        const double halo     = global_t[0];
        const double applyA   = global_t[1];
        const double axpy_u   = global_t[2];
        const double axpy_r   = global_t[3];
        const double update_p = global_t[4];
        const double dot      = global_t[5];
        const double dot_init = global_t[6];
        const double dot_pAp  = global_t[7];
        const double dot_rr   = global_t[8];
        const double cg_total = global_t[9];

        const bool file_exists = std::filesystem::exists(csv_path);
        std::ofstream ofs(csv_path, std::ios::app);
        if (!ofs) {
            std::cerr << "Error: could not open CSV file \""
                      << csv_path << "\" for writing.\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        if (!file_exists) {
            ofs << "N,nprocs,tol,max_iters,mode,converged,iters,"
                   "res0,res_final,rel_final,l2_error,max_error,runtime_sec,"
                   "t_halo,t_applyA,t_axpy_u,t_axpy_r,t_update_p,"
                   "t_dot,t_dot_init,t_dot_pAp,t_dot_rr,t_cg_total\n";
        }

        ofs << Nx << ","
            << nprocs << ","
            << std::setprecision(12) << tolerance << ","
            << max_iters << ","
            << modeToString(mode) << ","
            << (cg_res.converged ? 1 : 0) << ","
            << cg_res.iters << ","
            << cg_res.norm0 << ","
            << cg_res.norm_final << ","
            << rel_final << ","
            << l2_err << ","
            << max_err << ","
            << runtime_sec << ","
            << halo      << ","
            << applyA    << ","
            << axpy_u    << ","
            << axpy_r    << ","
            << update_p  << ","
            << dot       << ","
            << dot_init  << ","
            << dot_pAp   << ","
            << dot_rr    << ","
            << cg_total  << "\n";
    }

    // Also gather dot breakdown (GPU vs memcpy vs allreduce)
    double local_dot[3]  = {
        dot_times.t_gpu,
        dot_times.t_memcpy,
        dot_times.t_allreduce
    };
    double global_dot[3] = {0.0, 0.0, 0.0};

    MPI_Reduce(local_dot, global_dot, 3, MPI_DOUBLE, MPI_MAX,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Dot breakdown (max over ranks):\n"
                  << "  GPU work   = " << global_dot[0] << " s\n"
                  << "  memcpy     = " << global_dot[1] << " s\n"
                  << "  Allreduce  = " << global_dot[2] << " s\n";
    }

    MPI_Finalize();
    return 0;
}