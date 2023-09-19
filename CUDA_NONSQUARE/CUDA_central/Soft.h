//
//   Copyright (c) 2023, Yoshihiko Nishikawa, Werner Krauth, and A. C. Maggs
//
//   CUDA code for massively parallelized Monte Carlo simulation of
//   two-dimensional disks
//
//   URL: https://github.com/jellyfysh/SoftDisks
//   See LICENSE for copyright information
//
//   If you use this code or find it useful, please cite the following paper:
//
//   @article{PhysRevE.108.024103,
//       title = {Liquid-hexatic transition for soft disks},
//       author = {Nishikawa, Yoshihiko and Krauth, Werner and Maggs, A. C.},
//       journal = {Phys. Rev. E},
//       volume = {108},
//       issue = {2},
//       pages = {024103},
//       numpages = {7},
//       year = {2023},
//       month = {Aug},
//       publisher = {American Physical Society},
//       doi = {10.1103/PhysRevE.108.024103},
//       url = {https://link.aps.org/doi/10.1103/PhysRevE.108.024103}
//   }
//
//
#ifndef _Soft
#define _Soft

#include <algorithm>
#include <assert.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#include "../version.h" // this is the git string

// set automatically
//#define NumSAMPLES 1 // Number of independent runs
#define NNN (Nrow * Nrow) // number of particles
#define RAND              // the initial configuration: RANDOM
#define N_config 1000     // The number of configurations produced during a run
#define INPUT_DENSITY // This makes DENSITY/BETA input when the job is submitted
#define Nblock ((Nrow / 2))
#define nmax (10) //<- maximum number of particles in each block
#define BLOCKL                                                                 \
  32 //(Nblock / 2) // BLOCKL^2 / 4 is the number of threads in each block used
     // in GPU
#define GRIDL ((Nblock / BLOCKL) > 1 ? Nblock / BLOCKL : 1)
#define SHIFT_BLOCKL (16) //(Nblock / 4)
#define SHIFT_GRIDL ((Nblock > SHIFT_BLOCKL) ? Nblock / SHIFT_BLOCKL : 1)
#define MBLOCKL (32)
#define MGRIDL ((Nblock > MBLOCKL) ? Nblock / MBLOCKL : 1)
#define BLOCKS ((GRIDL * GRIDL))
#define THREADS ((BLOCKL * BLOCKL) / 4)
#define NCURAND ((BLOCKS > 200) ? 200 : BLOCKS)
#define nloop 16

// macros
#define rx(x, y, i)                                                            \
  rx[(i * Nblock + y) * Nblock + (x % 2 == 0 ? x / 2 : (x + Nblock) / 2)]
#define ry(x, y, i)                                                            \
  ry[(i * Nblock + y) * Nblock + (x % 2 == 0 ? x / 2 : (x + Nblock) / 2)]
#define rrx(x, y, i)                                                           \
  rx[(i * Nblock + y) * Nblock + (x % 2 == 0 ? x / 2 : (x + Nblock) / 2)]
#define rry(x, y, i)                                                           \
  ry[(i * Nblock + y) * Nblock + (x % 2 == 0 ? x / 2 : (x + Nblock) / 2)]
#define nparticle(x, y)                                                        \
  nparticle[(y * Nblock + (x % 2 == 0 ? x / 2 : (x + Nblock) / 2))]
#define devrx(x, y, i)                                                         \
  devrx[(i * Nblock + y) * Nblock + (x % 2 == 0 ? x / 2 : (x + Nblock) / 2)]
#define devry(x, y, i)                                                         \
  devry[(i * Nblock + y) * Nblock + (x % 2 == 0 ? x / 2 : (x + Nblock) / 2)]
#define nnparticle(x, y) nnparticle[(y * Nblock + x)]
#define temp_rx(x, y, i)                                                       \
  temp_rx[(i * Nblock + y) * Nblock + (x % 2 == 0 ? x / 2 : (x + Nblock) / 2)]
#define temp_ry(x, y, i)                                                       \
  temp_ry[(i * Nblock + y) * Nblock + (x % 2 == 0 ? x / 2 : (x + Nblock) / 2)]
#define temp_nparticle(x, y)                                                   \
  temp_nparticle[(y * Nblock + (x % 2 == 0 ? x / 2 : (x + Nblock) / 2))]

#define DUMMY_OFFSET -10.

using namespace std;

#if defined(SOFTCUDA)
void measure_e_p(double &, double &, double &);
#elif defined(HARDCUDA)
void ecmc_measure_pressure(double &);
void move_block(int, int, int);
void index_shift(int, int, int);
void measure_pressure(double &, double &);
void find_first_collision(vector<vector<int>> &, vector<vector<double>> &);
#endif
void check_particle_number(void);
void Set_Mersenne_Twister_GPU(int);
void Set_init_conf(void);
void random_config(void);
int reload(void);
void simulate(float);
void shift_checkerboard_CPU(float, float);

extern float density;
extern double Lbox, Lboy;
extern double Lblockx, Lblocky;
extern float *rx, *ry;
extern short *nparticle;
extern float *devrx, *devry;
extern short *devnparticle;
extern float rshiftx, rshifty;
extern float BETA;

extern curandStateMtgp32 (*devMTGPStates)[NCURAND];
extern mtgp32_kernel_params *devKernelParams;

extern std::random_device rd;
extern std::mt19937 mt;
extern std::uniform_real_distribution<double> genrand;

#if defined(SOFTCUDA) || defined(HARDCUDA)
extern texture<float, 1, cudaReadModeElementType> read_rx;
extern texture<float, 1, cudaReadModeElementType> read_ry;
extern texture<short, 1, cudaReadModeElementType> read_nparticle;

extern __global__ void random_generate(curandStateMtgp32 (*state)[NCURAND]);
extern __global__ void JUST_MOVE_f(float *, float *, short *, float *, float *,
                                   short *);
extern __global__ void update_Metropolis(float *, float *, unsigned int, float,
                                         float, float, float, double, float,
                                         curandStateMtgp32 (*state)[NCURAND]);
extern __global__ void shift_checkerboard(float *, float *, short *, float, int,
                                          double);
#endif

#endif
