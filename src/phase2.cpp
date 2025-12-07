/**
 * APMA 2822B Final Project
 * Phase 2: Distributed-memory (MPI) CPU Conjugate Gradient Solver
 *
 * Solves -Δu = f on the unit square with homogeneous Dirichlet boundary
 * conditions using a 5-point finite-difference stencil. The domain is
 * decomposed into horizontal slabs, one per MPI rank, with one halo row
 * above and below for ghost cells.
 *
 * Build modes:
 *   SingleMode  : single sine eigenmode with known exact solution
 *   MultiMode   : linear combination of several eigenmodes
 *   GenericRHS  : Gaussian bump RHS; no exact solution available
 *
 * Usage:
 *   mpirun -np P ./phase2 [Nx] [tol] [max_iters] [BuildMode] [csv_path]
 *
 *   Nx        : grid size in each dimension (Nx x Nx grid, default 1024)
 *   tol       : relative residual tolerance (default 1e-8)
 *   max_iters : maximum CG iterations (default 10000)
 *   BuildMode : 0 = SingleMode, 1 = MultiMode, 2 = GenericRHS
 *   csv_path  : output CSV path (default "results_phase2.csv")
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

// Map (i, j) to 1D index into a flattened Ny-major array
inline int idx(int i, int j, int Ny) {
    return i * Ny + j;
}

// True if (i, j) lies on the physical boundary of the global Nx x Ny grid
inline bool is_boundary(int i, int j, int Nx, int Ny) {
    return (i == 0 || i == Nx - 1 || j == 0 || j == Ny - 1);
}

// ============================================================================
// RHS and exact solution construction
// ============================================================================

// Assemble right-hand side f and exact solution u (if available) on this rank's slab
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

            // Dirichlet boundary: u = 0, f = 0
            if (is_boundary(i_global, j, Nx, Ny)) {
                exact[index] = 0.0;
                rhs[index]   = 0.0;
                continue;
            }

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
// Error metrics
// ============================================================================

// Compute global L2 and max error ||u - exact|| using all ranks
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
// Stencil application
// ============================================================================

// Apply 5-point Laplacian: y = A x on local slab, using halo rows.
// We loop only over local interior rows 1..owned. Column boundaries
// j=0 and j=Ny-1 are Dirichlet (y=0).
void apply_A(const std::vector<double>& x,
             std::vector<double>&       y,
             int owned,
             int Ny,
             double inv_h2)
{
    for (int i = 1; i <= owned; ++i) {
        for (int j = 0; j < Ny; ++j) {
            const int index = idx(i, j, Ny);

            // Physical left/right boundaries
            if (j == 0 || j == Ny - 1) {
                y[index] = 0.0;
                continue;
            }

            const double center = x[index];
            const double up     = x[idx(i - 1, j,     Ny)];
            const double down   = x[idx(i + 1, j,     Ny)];
            const double left   = x[idx(i,     j - 1, Ny)];
            const double right  = x[idx(i,     j + 1, Ny)];

            y[index] = (4.0 * center - up - down - left - right) * inv_h2;
        }
    }
}

// ============================================================================
// Basic linear algebra helpers (MPI-aware)
// ============================================================================

// Global dot product over interior rows 1..owned on each rank
double dot(const std::vector<double>& a,
           const std::vector<double>& b,
           int owned,
           int Ny)
{
    double local_sum  = 0.0;
    double global_sum = 0.0;

    for (int i = 1; i <= owned; ++i) {
        for (int j = 0; j < Ny; ++j) {
            const int index = idx(i, j, Ny);
            local_sum += a[index] * b[index];
        }
    }

    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return global_sum;
}

// y <- y + alpha * x  (on full local vector including halos)
void axpy(double alpha,
          const std::vector<double>& x,
          std::vector<double>&       y)
{
    const std::size_t n = y.size();
    for (std::size_t i = 0; i < n; ++i) {
        y[i] += alpha * x[i];
    }
}

// Exchange top/bottom halo rows with neighbors using non-blocking sends/receives
inline void exchange_halos(std::vector<double>& u,
                           int Ny,
                           int owned,
                           int top,
                           int bottom)
{
    double* top_halo    = u.data();                    // row 0
    double* first_owned = u.data() + Ny;               // row 1
    double* last_owned  = u.data() + owned * Ny;       // row owned
    double* bottom_halo = u.data() + (owned + 1) * Ny; // row owned+1

    MPI_Request reqs[4];
    int r = 0;

    MPI_Irecv(top_halo,    Ny, MPI_DOUBLE, top,    0, MPI_COMM_WORLD, &reqs[r++]);
    MPI_Irecv(bottom_halo, Ny, MPI_DOUBLE, bottom, 1, MPI_COMM_WORLD, &reqs[r++]);

    MPI_Isend(first_owned, Ny, MPI_DOUBLE, top,    1, MPI_COMM_WORLD, &reqs[r++]);
    MPI_Isend(last_owned,  Ny, MPI_DOUBLE, bottom, 0, MPI_COMM_WORLD, &reqs[r++]);

    MPI_Waitall(r, reqs, MPI_STATUSES_IGNORE);

    if (top == MPI_PROC_NULL) {
        std::fill(top_halo, top_halo + Ny, 0.0);
    }
    if (bottom == MPI_PROC_NULL) {
        std::fill(bottom_halo, bottom_halo + Ny, 0.0);
    }
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
// Conjugate Gradient solver (MPI, CPU)
// ============================================================================

void cg_solve(int Nx,
              int Ny,
              int local_size,    // size of local vectors (including halo rows)
              int owned,         // interior rows owned by this rank
              int top,
              int bottom,
              int rank,
              double h,
              const std::vector<double>& rhs,
              std::vector<double>&       u,
              int max_iters,
              double tol,
              CGResult& result)
{
    (void)Nx; // Nx is implicit in owned/local_size, but kept for symmetry

    const double h2     = h * h;
    const double inv_h2 = 1.0 / h2;

    std::vector<double> r(local_size, 0.0);
    std::vector<double> p(local_size, 0.0);
    std::vector<double> Ap(local_size, 0.0);

    // Initial guess: u = 0. So r = b - A u = b
    r = rhs;
    p = r;

    double rho   = dot(r, r, owned, Ny);
    double norm0 = std::sqrt(rho);
    result.norm0 = norm0;

    if (norm0 < tol) {
        // rhs is effectively zero; nothing to do
        result.iters      = 0;
        result.norm_final = norm0;
        result.converged  = true;
        return;
    }

    for (int k = 0; k < max_iters; ++k) {

        // 1. Halo exchange for p
        exchange_halos(p, Ny, owned, top, bottom);

        // 2. Ap = A p
        apply_A(p, Ap, owned, Ny, inv_h2);

        // 3. alpha = (r, r) / (p, Ap)
        const double p_dot_Ap = dot(p, Ap, owned, Ny);
        if (p_dot_Ap <= 0.0) {
            if (rank == 0) {
                std::cerr << "CG breakdown: p^T A p <= 0 at iter " << k << "\n";
            }
            break;
        }

        const double alpha = rho / p_dot_Ap;

        // 4. u = u + alpha * p
        axpy(alpha, p, u);

        // 5. r = r - alpha * Ap
        axpy(-alpha, Ap, r);

        // 6. new_rho = (r, r), check convergence
        const double new_rho = dot(r, r, owned, Ny);
        const double norm    = std::sqrt(new_rho);
        const double rel     = norm / norm0;

        if (rel < tol) {
            if (rank == 0) {
                std::cout << "converged in " << (k + 1)
                          << " iterations, rel = " << rel << "\n";
            }
            result.iters      = k + 1;
            result.norm_final = norm;
            result.converged  = true;
            return;
        }

        // 7. p = r + beta * p
        const double beta = new_rho / rho;
        for (int j = 0; j < local_size; ++j) {
            p[j] = r[j] + beta * p[j];
        }

        rho               = new_rho;
        result.iters      = k + 1;
        result.norm_final = norm;
    }

    // If we exit the loop without hitting the convergence criterion:
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
    std::string csv_path = "results_phase2.csv";

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

    // 1D slab decomposition in the x-direction (rows)
    const int N_int      = Nx - 2;         // number of global interior rows
    const int base       = N_int / nprocs;
    const int rem        = N_int % nprocs;

    const int owned      = base + (rank < rem ? 1 : 0); // interior rows on this rank
    const int start_off  = rank * base + std::min(rank, rem);

    const int local_rows = owned + 2;     // interior rows + 2 halo rows
    const int local_size = local_rows * Ny;

    const int top        = (rank == 0)          ? MPI_PROC_NULL : rank - 1;
    const int bottom     = (rank == nprocs - 1) ? MPI_PROC_NULL : rank + 1;

    const int global_start = 1 + start_off; // first global interior row index

    // Grid spacing
    const double h = 1.0 / (Nx - 1);

    // Local vectors (include halos)
    std::vector<double> u(local_size,   0.0);
    std::vector<double> rhs(local_size, 0.0);
    std::vector<double> exact(local_size, 0.0);

    // Assemble RHS and exact solution (if available)
    build(Nx, Ny, global_start, owned, h, rhs, exact, mode);

    // Run CG and time it (using MPI_Wtime and global max time)
    CGResult cg_res;

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();
    cg_solve(Nx, Ny, local_size, owned, top, bottom, rank, h,
             rhs, u, max_iters, tolerance, cg_res);
    const double t1 = MPI_Wtime();

    const double local_time  = t1 - t0;
    double       runtime_sec = 0.0;

    MPI_Reduce(&local_time, &runtime_sec, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

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
            std::cout << "L2 error   = " << l2_err  << "\n"
                      << "Max error  = " << max_err << "\n";
        }
    }

    // Final relative residual for CSV
    double rel_final = 0.0;
    if (cg_res.norm0 > 0.0) {
        rel_final = cg_res.norm_final / cg_res.norm0;
    }

    // Append summary row to CSV on rank 0
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