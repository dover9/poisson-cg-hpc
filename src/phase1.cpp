/**
 * APMA 2822B Final Project
 * Phase 1: Serial CPU Conjugate Gradient Solver
 *
 * Solves -Δu = f on the unit square with homogeneous Dirichlet boundary
 * conditions using a 5-point finite-difference stencil. This phase runs
 * a serial Conjugate Gradient (CG) solver on the CPU and writes summary
 * statistics to a CSV file for post-processing.
 *
 * Build modes:
 *   SingleMode  : single sine eigenmode with known exact solution
 *   MultiMode   : linear combination of several eigenmodes
 *   GenericRHS  : Gaussian bump RHS; no exact solution available
 *
 * Usage:
 *   ./phase1 [Nx] [tol] [max_iters] [BuildMode] [csv_path]
 *
 *   Nx        : grid size in each dimension (Nx x Nx grid, default 1024)
 *   tol       : relative residual tolerance (default 1e-8)
 *   max_iters : maximum CG iterations (default 10000)
 *   BuildMode : 0 = SingleMode, 1 = MultiMode, 2 = GenericRHS
 *   csv_path  : output CSV path (default "results_phase1.csv")
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

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

// Assemble right-hand side f and exact solution u (if available). Own interior 
// rows 1..Nx-2, and treat row 0 and Nx-1 as Dirichlet boundary (u = 0, f = 0).
void build(int Nx, int Ny,
           int global_start,   // first global interior row owned (1 in serial)
           int owned,          // number of interior rows (Nx - 2 in serial)
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

// Compute L2 and max error ||u - exact|| on the serial grid
void compute_error_serial(const std::vector<double>& u,
                          const std::vector<double>& exact,
                          int Nx,
                          int Ny,
                          double& l2_err,
                          double& max_err)
{
    double sum_sq = 0.0;
    double max_val = 0.0;

    // Only interior rows (1..Nx-2), all columns
    for (int i = 1; i <= Nx - 2; ++i) {
        for (int j = 0; j < Ny; ++j) {
            const int k = idx(i, j, Ny);
            const double diff = std::abs(u[k] - exact[k]);
            sum_sq += diff * diff;
            max_val = std::max(max_val, diff);
        }
    }

    l2_err  = std::sqrt(sum_sq);
    max_err = max_val;
}

// ============================================================================
// CPU linear algebra helpers
// ============================================================================

// Apply 5-point Laplacian: y = A x on the full Nx x Ny grid
// Dirichlet boundaries are enforced as u = 0.
void apply_A_cpu(const std::vector<double>& x,
                 std::vector<double>&       y,
                 int Nx,
                 int Ny,
                 double inv_h2)
{
    // Zero out y first (for safety)
    std::fill(y.begin(), y.end(), 0.0);

    for (int i = 1; i <= Nx - 2; ++i) {
        for (int j = 1; j <= Ny - 2; ++j) {
            const int index = idx(i, j, Ny);

            const double center = x[index];
            const double up     = x[idx(i - 1, j,     Ny)];
            const double down   = x[idx(i + 1, j,     Ny)];
            const double left   = x[idx(i,     j - 1, Ny)];
            const double right  = x[idx(i,     j + 1, Ny)];

            y[index] = (4.0 * center - up - down - left - right) * inv_h2;
        }
    }

    // Boundary rows/cols remain zero (Dirichlet BC)
}

// Dot product over full vector (boundaries are zero, so they don't contribute)
double dot(const std::vector<double>& a,
           const std::vector<double>& b)
{
    const int n = static_cast<int>(a.size());
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

// y <- y + alpha * x
void axpy(double alpha,
          const std::vector<double>& x,
          std::vector<double>&       y)
{
    const int n = static_cast<int>(x.size());
    for (int i = 0; i < n; ++i) {
        y[i] += alpha * x[i];
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
// Conjugate Gradient solver (serial CPU)
// ============================================================================

void cg_solve(int Nx,
              int Ny,
              double h,
              const std::vector<double>& rhs,
              std::vector<double>&       u,   // solution (Nx x Ny, includes boundaries)
              int max_iters,
              double tol,
              CGResult& result)
{
    const int    Ntot   = Nx * Ny;
    const double h2     = h * h;
    const double inv_h2 = 1.0 / h2;

    std::vector<double> r(Ntot, 0.0);
    std::vector<double> p(Ntot, 0.0);
    std::vector<double> Ap(Ntot, 0.0);

    // Initial guess: u = 0 (assumed by caller)
    // r = b - A u = b since u = 0
    r = rhs;
    p = r;

    double rho   = dot(r, r);
    double norm0 = std::sqrt(rho);
    result.norm0 = norm0;

    if (norm0 < tol) {
        result.iters      = 0;
        result.norm_final = norm0;
        result.converged  = true;
        return;
    }

    for (int k = 0; k < max_iters; ++k) {
        // Ap = A p
        apply_A_cpu(p, Ap, Nx, Ny, inv_h2);

        // alpha = (r, r) / (p, Ap)
        const double p_dot_Ap = dot(p, Ap);
        if (p_dot_Ap <= 0.0) {
            std::cerr << "CG breakdown: p^T A p <= 0 at iter " << k << "\n";
            break;
        }

        const double alpha = rho / p_dot_Ap;

        // u = u + alpha * p
        axpy(alpha, p, u);

        // r = r - alpha * Ap
        axpy(-alpha, Ap, r);

        // Check convergence
        const double new_rho = dot(r, r);
        const double norm    = std::sqrt(new_rho);
        const double rel     = norm / norm0;

        if (rel < tol) {
            std::cout << "converged in " << (k + 1)
                      << " iterations, rel = " << rel << "\n";
            result.iters      = k + 1;
            result.norm_final = norm;
            result.converged  = true;
            return;
        }

        // p = r + beta * p
        const double beta = new_rho / rho;
        for (int i = 0; i < Ntot; ++i) {
            p[i] = r[i] + beta * p[i];
        }

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
    std::string csv_path = "results_phase1.csv";

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

    const int nprocs = 1;  // serial phase

    std::cout << "Nx=" << Nx
              << " tol=" << tolerance
              << " max_iters=" << max_iters
              << " mode=" << modeToString(mode)
              << " csv=\"" << csv_path << "\"\n";

    const int Nx_int = Nx - 2;    // number of interior rows
    const int Ny_int = Ny - 2;    // not used explicitly, but conceptually
    (void)Ny_int;                 // silence unused variable warning if any

    const int owned        = Nx_int;   // interior rows 1..Nx-2
    const int global_start = 1;        // first interior row index

    const int Ntot = Nx * Ny;
    const double h = 1.0 / (Nx - 1);

    // Host vectors (including boundaries)
    std::vector<double> u(Ntot, 0.0);
    std::vector<double> rhs(Ntot, 0.0);
    std::vector<double> exact(Ntot, 0.0);

    // Assemble RHS and exact solution (if available)
    build(Nx, Ny, global_start, owned, h, rhs, exact, mode);

    // Run CG and time it
    CGResult cg_res;

    auto t0 = std::chrono::high_resolution_clock::now();
    cg_solve(Nx, Ny, h, rhs, u, max_iters, tolerance, cg_res);
    auto t1 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = t1 - t0;
    const double runtime_sec = elapsed.count();

    // Error metrics (only meaningful when exact solution is known)
    double l2_err  = -1.0;
    double max_err = -1.0;

    if (mode == GenericRHS) {
        std::cout << "GenericRHS mode: skipping exact error check.\n";
    } else {
        compute_error_serial(u, exact, Nx, Ny, l2_err, max_err);
        std::cout << "L2 error   = "  << l2_err  << "\n"
                  << "Max error  = "  << max_err << "\n";
    }

    // Final relative residual
    double rel_final = 0.0;
    if (cg_res.norm0 > 0.0) {
        rel_final = cg_res.norm_final / cg_res.norm0;
    }

    // Append summary row to CSV
    {
        const bool file_exists = std::filesystem::exists(csv_path);
        std::ofstream ofs(csv_path, std::ios::app);
        if (!ofs) {
            std::cerr << "Error: could not open CSV file \""
                      << csv_path << "\" for writing.\n";
            return 1;
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

    return 0;
}