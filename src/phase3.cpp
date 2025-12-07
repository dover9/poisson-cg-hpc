/**
 * APMA 2822B Final Project
 * Phase 3: GPU-Accelerated Conjugate Gradient (MPI + HIP)
 *
 * Solves -Δu = f on the unit square with homogeneous Dirichlet boundary
 * conditions using a 5-point finite-difference stencil. The global domain
 * is decomposed into horizontal slabs across MPI ranks. Each rank owns a
 * contiguous set of interior rows and exchanges halo rows with its neighbors.
 *
 * This phase offloads the CG linear algebra (apply_A, axpy, dot, update_p)
 * to a single GPU per rank using HIP. Dot products are computed on the GPU
 * and then reduced globally with MPI_Allreduce.
 *
 * Build modes:
 *   SingleMode  : single sine eigenmode with known exact solution
 *   MultiMode   : linear combination of several eigenmodes
 *   GenericRHS  : Gaussian bump RHS; no exact solution available
 *
 * Usage:
 *   mpirun -np P ./phase3 [Nx] [tol] [max_iters] [BuildMode] [csv_path]
 *
 *   Nx        : grid size in each dimension (Nx x Nx grid, default 1024)
 *   tol       : relative residual tolerance (default 1e-8)
 *   max_iters : maximum CG iterations (default 10000)
 *   BuildMode : 0 = SingleMode, 1 = MultiMode, 2 = GenericRHS
 *   csv_path  : output CSV path (default "results_phase3.csv")
 */

#include <algorithm>
#include <cmath>
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
// Problem configuration
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

// Map (i, j) to 1D index into a flattened Ny-major array: [0..Nx-1] x [0..Ny-1]
__host__ __device__
int idx(int i, int j, int Ny) {
    return i * Ny + j;
}

// True if (i, j) lies on the physical boundary of the global Nx x Ny grid
bool is_boundary(int i, int j, int Nx, int Ny) {
    return (i == 0 || i == Nx - 1 || j == 0 || j == Ny - 1);
}

// ============================================================================
// RHS and exact solution construction
// ============================================================================

// Assemble right-hand side f and exact solution u (if available) on this rank's slab
void build(int Nx, int Ny,
           int global_start,   // first global interior row owned by this rank
           int owned,          // number of interior rows on this rank
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

            // Dirichlet boundary: u = 0, f = 0
            if (is_boundary(i_global, j, Nx, Ny)) {
                exact[index] = 0.0;
                rhs[index]   = 0.0;
                continue;
            }

            const double y  = j * h;
            const double u1 = std::sin(PI * x) * std::sin(PI * y);   // (1,1), λ1 = 2π²
            const double f1 = (2.0 * PI * PI) * u1;                  // -Δ u1 = 2π² u1

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
                    exact[index] = 0.0;  // unknown exact solution
                    break;
                }
            }
        }
    }
}

// ============================================================================
// Error metrics
// ============================================================================

// Compute global L2 and max error ||u - exact|| using MPI_Allreduce
void compute_error_parallel(const std::vector<double>& u,
                            const std::vector<double>& exact,
                            int owned,
                            int Ny,
                            double& l2_err,
                            double& max_err)
{
    double local_sum_sq = 0.0;
    double local_max    = 0.0;

    // Only interior rows (1..owned), all columns
    for (int i = 1; i <= owned; ++i) {
        for (int j = 0; j < Ny; ++j) {
            const int k = idx(i, j, Ny);
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
// MPI halo exchange
// ============================================================================

// Exchange top/bottom halo rows of u with neighbors (Dirichlet BC at domain edges)
static inline void exchange_halos(std::vector<double>& u,
                                  int Ny,
                                  int owned,
                                  int top,
                                  int bottom)
{
    double* top_halo    = u.data();                 // row 0
    double* first_owned = u.data() + Ny;            // row 1
    double* last_owned  = u.data() + owned * Ny;    // row owned
    double* bottom_halo = u.data() + (owned + 1) * Ny; // row owned+1

    MPI_Request reqs[4];
    int r = 0;

    MPI_Irecv(top_halo,    Ny, MPI_DOUBLE, top,    0, MPI_COMM_WORLD, &reqs[r++]);
    MPI_Irecv(bottom_halo, Ny, MPI_DOUBLE, bottom, 1, MPI_COMM_WORLD, &reqs[r++]);
    MPI_Isend(first_owned, Ny, MPI_DOUBLE, top,    1, MPI_COMM_WORLD, &reqs[r++]);
    MPI_Isend(last_owned,  Ny, MPI_DOUBLE, bottom, 0, MPI_COMM_WORLD, &reqs[r++]);

    MPI_Waitall(r, reqs, MPI_STATUSES_IGNORE);

    // Apply Dirichlet BC at domain boundaries
    if (top    == MPI_PROC_NULL) std::fill(top_halo,    top_halo    + Ny, 0.0);
    if (bottom == MPI_PROC_NULL) std::fill(bottom_halo, bottom_halo + Ny, 0.0);
}

// ============================================================================
// CG statistics
// ============================================================================

struct CGResult {
    int    iters      = 0;
    double norm0      = 0.0;
    double norm_final = 0.0;
    bool   converged  = false;
};

// ============================================================================
// Device buffer manager
// ============================================================================

struct DeviceBuffers {
    double* d_u   = nullptr;
    double* d_r   = nullptr;
    double* d_p   = nullptr;
    double* d_Ap  = nullptr;
    double* d_buf = nullptr;   // scratch for dot products (partials + final)

    int     n     = 0;

    void allocate(int n_local) {
        n = n_local;
        HIP_CHECK(hipMalloc(&d_u,   n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_r,   n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_p,   n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_Ap,  n * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_buf, n * sizeof(double))); // used for block partials + result
    }

    void free() {
        if (d_u)   HIP_CHECK(hipFree(d_u));
        if (d_r)   HIP_CHECK(hipFree(d_r));
        if (d_p)   HIP_CHECK(hipFree(d_p));
        if (d_Ap)  HIP_CHECK(hipFree(d_Ap));
        if (d_buf) HIP_CHECK(hipFree(d_buf));
        d_u = d_r = d_p = d_Ap = d_buf = nullptr;
        n = 0;
    }

    ~DeviceBuffers() {
        free();
    }
};

// ============================================================================
// GPU kernels
// ============================================================================

// Apply 5-point Laplacian: y = A x on local slab (including halos in x)
__global__
void apply_A_kernel(const double* __restrict__ x,
                    double* __restrict__ y,
                    int owned,
                    int Ny,
                    double inv_h2)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1; // interior rows: 1..owned
    int j = blockIdx.x * blockDim.x + threadIdx.x;     // columns: 0..Ny-1

    if (i < 1 || i > owned || j < 0 || j >= Ny) {
        return;
    }

    const int index = idx(i, j, Ny);

    // Left/right physical boundaries are enforced as u = 0
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

// y <- y + alpha * x
__global__
void axpy_kernel(int n, double alpha,
                 const double* __restrict__ x,
                 double*       __restrict__ y)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) {
        y[k] += alpha * x[k];
    }
}

// p <- r + beta * p
__global__
void update_p_kernel(int n, double beta,
                     const double* __restrict__ r,
                     double*       __restrict__ p)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) {
        p[k] = r[k] + beta * p[k];
    }
}

// First stage: per-block partial dot products into partial[blockIdx.x]
__global__
void dot_partial_kernel(const double* __restrict__ a,
                        const double* __restrict__ b,
                        double*       __restrict__ partial,
                        int n)
{
    extern __shared__ double sdata[];

    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + threadIdx.x;

    double val = 0.0;
    if (i < n) {
        val = a[i] * b[i];
    }

    sdata[tid] = val;
    __syncthreads();

    // Intra-block reduction to sdata[0]
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

// Second stage: reduce partial[0..num_blocks-1] into result[0]
__global__
void dot_reduce_kernel(const double* __restrict__ partial,
                       double*       __restrict__ result,
                       int num_blocks)
{
    extern __shared__ double sdata[];

    int tid  = threadIdx.x;
    double sum = 0.0;

    // Each thread accumulates a strided chunk of partial[]
    for (int i = tid; i < num_blocks; i += blockDim.x) {
        sum += partial[i];
    }

    sdata[tid] = sum;
    __syncthreads();

    // Reduce within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        result[0] = sdata[0];
    }
}

// ============================================================================
// GPU wrappers
// ============================================================================

// Apply A on device slab
void apply_A(double* d_x, double* d_y, int owned, int Ny, double inv_h2)
{
    dim3 block(32, 8);
    dim3 grid((Ny    + block.x - 1) / block.x,
              (owned + block.y - 1) / block.y);

    apply_A_kernel<<<grid, block>>>(d_x, d_y, owned, Ny, inv_h2);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
}

// Global dot product <a, b> using GPU partial sums + MPI_Allreduce
double dot(const double* d_a,
           const double* d_b,
           double*       d_buf,
           int           local_size)
{
    const int threads = 256;
    const int blocks  = (local_size + threads - 1) / threads;

    // Stage 1: per-block partial sums -> d_buf[0..blocks-1]
    size_t shmem = threads * sizeof(double);
    dot_partial_kernel<<<blocks, threads, shmem>>>(d_a, d_b, d_buf, local_size);
    HIP_CHECK(hipGetLastError());

    // Stage 2: reduce partials into d_buf[0]
    const int red_threads = 256;
    size_t red_shmem = red_threads * sizeof(double);
    dot_reduce_kernel<<<1, red_threads, red_shmem>>>(d_buf, d_buf, blocks);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    // Copy local sum to host
    double local_sum = 0.0;
    HIP_CHECK(hipMemcpy(&local_sum, d_buf, sizeof(double), hipMemcpyDeviceToHost));

    // Global reduction
    double global_sum = 0.0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return global_sum;
}

// ============================================================================
// Conjugate Gradient solver (MPI + GPU)
// ============================================================================

void cg_solve(int Nx, int Ny,
              int local_size,
              int local_rows,
              int owned,
              int top,
              int bottom,
              int rank,
              double h,
              std::vector<double>& u,   // host solution (includes halos)
              int max_iters,
              double tol,
              CGResult& result,
              DeviceBuffers& dev)
{
    const double h2     = h * h;
    const double inv_h2 = 1.0 / h2;

    // Host buffer for halo exchange of search direction p
    std::vector<double> p_host(local_size, 0.0);

    // Initial residual: r = b - A u; u is zero, and dev.d_r initially holds b
    double rho   = dot(dev.d_r, dev.d_r, dev.d_buf, local_size);
    double norm0 = std::sqrt(rho);
    result.norm0 = norm0;

    if (norm0 < tol) {
        // RHS is effectively zero; nothing to do
        result.iters      = 0;
        result.norm_final = norm0;
        result.converged  = true;

        HIP_CHECK(hipMemcpy(u.data(), dev.d_u, local_size * sizeof(double),
                            hipMemcpyDeviceToHost));
        return;
    }

    const int threads = 256;
    const int blocks  = (local_size + threads - 1) / threads;

    for (int k = 0; k < max_iters; ++k) {
        // 1. Halo exchange for p (host)
        HIP_CHECK(hipMemcpy(p_host.data(), dev.d_p,
                            local_size * sizeof(double),
                            hipMemcpyDeviceToHost));
        exchange_halos(p_host, Ny, owned, top, bottom);
        HIP_CHECK(hipMemcpy(dev.d_p, p_host.data(),
                            local_size * sizeof(double),
                            hipMemcpyHostToDevice));

        // 2. Ap = A p
        apply_A(dev.d_p, dev.d_Ap, owned, Ny, inv_h2);

        // 3. alpha = (r, r) / (p, Ap)
        double p_dot_Ap = dot(dev.d_p, dev.d_Ap, dev.d_buf, local_size);
        if (p_dot_Ap <= 0.0) {
            if (rank == 0) {
                std::cerr << "CG breakdown: p^T A p <= 0 at iter " << k << "\n";
            }
            break;
        }

        const double alpha = rho / p_dot_Ap;

        // 4. u = u + alpha * p
        axpy_kernel<<<blocks, threads>>>(local_size, alpha, dev.d_p, dev.d_u);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // 5. r = r - alpha * Ap
        axpy_kernel<<<blocks, threads>>>(local_size, -alpha, dev.d_Ap, dev.d_r);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // 6. new_rho = (r, r); check convergence
        double new_rho = dot(dev.d_r, dev.d_r, dev.d_buf, local_size);
        double norm    = std::sqrt(new_rho);
        double rel     = norm / norm0;

        if (rel < tol) {
            if (rank == 0) {
                std::cout << "converged in " << (k + 1)
                          << " iterations, rel = " << rel << "\n";
            }
            result.iters      = k + 1;
            result.norm_final = norm;
            result.converged  = true;

            HIP_CHECK(hipMemcpy(u.data(), dev.d_u,
                                local_size * sizeof(double),
                                hipMemcpyDeviceToHost));
            return;
        }

        // 7. p = r + beta * p
        const double beta = new_rho / rho;
        update_p_kernel<<<blocks, threads>>>(local_size, beta, dev.d_r, dev.d_p);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        rho               = new_rho;
        result.iters      = k + 1;
        result.norm_final = norm;
    }

    // If we exit the loop without satisfying tolerance
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
    std::string csv_path = "results_phase3.csv";

    if (argc > 1) { Nx = Ny = std::stoi(argv[1]); }
    if (argc > 2) { tolerance = std::stod(argv[2]); }
    if (argc > 3) { max_iters = std::stoi(argv[3]); }
    if (argc > 4) {
        int m = std::stoi(argv[4]);
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
        std::cout << "Nx=" << Nx
                  << " tol=" << tolerance
                  << " max_iters=" << max_iters
                  << " mode=" << modeToString(mode)
                  << " csv=\"" << csv_path << "\"\n";
    }

    // One GPU per rank (round-robin across devices)
    int device_count = 0;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    int device_id = (device_count > 0 ? (rank % device_count) : 0);
    HIP_CHECK(hipSetDevice(device_id));

    // 1D slab decomposition in x-direction
    const int N_int      = Nx - 2;                    // number of interior rows globally
    const int base       = N_int / nprocs;
    const int rem        = N_int % nprocs;
    const int owned      = base + (rank < rem ? 1 : 0);
    const int start_off  = rank * base + std::min(rank, rem);
    const int local_rows = owned + 2;                 // include top/bottom halos
    const int local_size = local_rows * Ny;

    const int top    = (rank == 0)          ? MPI_PROC_NULL : rank - 1;
    const int bottom = (rank == nprocs - 1) ? MPI_PROC_NULL : rank + 1;

    const int global_start = 1 + start_off;  // first global interior row index

    // Host vectors (including halos)
    std::vector<double> u(local_size, 0.0);
    std::vector<double> rhs(local_size, 0.0);
    std::vector<double> exact(local_size, 0.0);

    const double h = 1.0 / (Nx - 1);
    build(Nx, Ny, global_start, owned, h, rhs, exact, mode);

    // Device buffers
    DeviceBuffers dev;
    dev.allocate(local_size);

    // Initialize device fields: u = 0, r = b, p = r
    HIP_CHECK(hipMemset(dev.d_u, 0, local_size * sizeof(double)));
    HIP_CHECK(hipMemcpy(dev.d_r, rhs.data(),
                        local_size * sizeof(double),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dev.d_p, dev.d_r,
                        local_size * sizeof(double),
                        hipMemcpyDeviceToDevice));

    // Run CG and time it
    CGResult cg_res;
    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    cg_solve(Nx, Ny, local_size, local_rows, owned,
             top, bottom, rank, h,
             u, max_iters, tolerance,
             cg_res, dev);

    const double t1         = MPI_Wtime();
    const double local_time = t1 - t0;
    double       runtime_sec = 0.0;

    // Use max over ranks as global runtime
    MPI_Reduce(&local_time, &runtime_sec, 1, MPI_DOUBLE,
               MPI_MAX, 0, MPI_COMM_WORLD);

    // Error metrics (only meaningful when exact solution is known)
    double l2_err  = -1.0;
    double max_err = -1.0;

    if (mode == GenericRHS) {
        if (rank == 0) {
            std::cout << "GenericRHS mode: skipping exact error check.\n";
        }
    } else {
        compute_error_parallel(u, exact, owned, Ny, l2_err, max_err);
        if (rank == 0) {
            std::cout << "L2 error   = "  << l2_err  << "\n"
                      << "Max error  = "  << max_err << "\n";
        }
    }

    // Final relative residual
    double rel_final = 0.0;
    if (cg_res.norm0 > 0.0) {
        rel_final = cg_res.norm_final / cg_res.norm0;
    }

    // Rank 0: append summary row to CSV
    if (rank == 0) {
        const bool file_exists = std::filesystem::exists(csv_path);
        std::ofstream ofs(csv_path, std::ios::app);
        if (!ofs) {
            std::cerr << "Error: could not open CSV file \""
                      << csv_path << "\" for writing.\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        if (!file_exists) {
            ofs << "N,nprocs,tol,max_iters,mode,converged,iters,"
                   "res0,res_final,rel_final,l2_error,max_error,runtime_sec\n";
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
            << runtime_sec << "\n";
    }

    MPI_Finalize();
    return 0;
}