# Distributed Conjugate Gradient Solver (MPI + GPU, HIP)

This project implements a high-performance Conjugate Gradient (CG) solver for
the 2‑D Poisson equation using four progressively more advanced architectures:

1. **Phase 1 — Serial CPU**
2. **Phase 2 — MPI CPU**
3. **Phase 3 — MPI + GPU (HIP)**
4. **Phase 4 — Distributed MPI + GPU with detailed timing + roofline analysis**

The solver demonstrates how performance evolves when moving from a single‑core
implementation to a fully distributed GPU system. It also includes analysis
scripts for scaling, timing breakdowns, and roofline modeling.

---

## Problem Setup

We solve:

```
-Δu = f   on (0,1)²
u = 0     on ∂Ω
```

using the 5‑point finite‑difference Laplacian on an **N×N** grid.  
The resulting linear system **Au = b** is large, sparse, and SPD → ideal for CG.

### Build Modes

| Mode | Meaning |
|------|---------|
| 0 | Single eigenfunction (closed-form exact solution) |
| 1 | Mixture of several eigenfunctions |
| 2 | Generic Gaussian RHS (no exact solution) |

---

## Repository Structure

```
.
├── phase1.cpp
├── phase2.cpp
├── phase3.cpp
├── phase4.cpp
├── results/
├── plots/
├── analyze_cg.py
├── convergence.py
├── roofline_cg.py
├── utils.py
└── README.md
```

---

## Building

### **Phase 1 — CPU**
```
g++ -O3 -std=c++20 -DNDEBUG phase1.cpp -o phase1
```

### **Phase 2 — MPI CPU**
```
mpicxx -O3 -std=c++20 -DNDEBUG phase2.cpp -o phase2
```

### **Phase 3 — MPI + GPU (HIP)**
```
hipcc -DUSE_GPU -std=c++20 -O3 --offload-arch=gfx90a phase3.cpp -o phase3   $(mpicxx --showme:compile) $(mpicxx --showme:link)
```

### **Phase 4 — Distributed MPI + GPU**
```
hipcc -DUSE_GPU -std=c++20 -O3 --offload-arch=gfx90a phase4.cpp -o phase4   $(mpicxx --showme:compile) $(mpicxx --showme:link)
```

---

## Running

### **Phase 1**
```
for N in 257 513 1025 2049 4097 8193; do
  ./phase1 $N 1e-8 10000 2 results_phase1.csv
done
```

### **Phase 2**
```
for P in 1 2 4 8; do
  mpirun -np $P ./phase2 4097 1e-8 10000 2 results_phase2.csv
done
```

### **Phase 3 — SLURM Job**
```
#!/bin/bash
#SBATCH -J phase3_sweep
#SBATCH -o phase3.%j.out
#SBATCH -N 2
#SBATCH -n 8
#SBATCH -t 01:00:00
#SBATCH -p mi2104x

cd /home1/bustudent53
rm -f results_phase3.csv
module load rocm/7.1.0

for N in 257 513 1025 2049 4097; do
  for P in 1 2 4 8; do
    mpirun -np $P ./phase3 $N 1e-8 10000 2 results_phase3.csv
  done
done
```

### **Phase 4 — SLURM Job**
```
#!/bin/bash
#SBATCH -J phase4_sweep
#SBATCH -o phase4.%j.out
#SBATCH -N 2
#SBATCH -n 8
#SBATCH -t 01:00:00
#SBATCH -p mi2104x

cd /home1/bustudent53
rm -f results_phase4.csv
module load rocm/7.1.0

for N in 257 513 1025 2049 4097; do
  for P in 1 2 4 8; do
    mpirun -np $P ./phase4 $N 1e-8 10000 2 results_phase4.csv
  done
done
```

---

## Python Analysis

Run:

```
python3 convergence.py
python3 analyze_cg.py
python3 roofline_cg.py
```

---

## Notes

- Phase 4 assumes **GPU-aware MPI** (direct device-pointer halo exchange).
- CSV files **append** results — delete them before new runs.
- Solver uses **Jacobi-free CG**, ideal for a clean performance study.
- A natural extension is **multigrid preconditioning**.

---

## License

You may include the license of your choice (MIT recommended).

---

