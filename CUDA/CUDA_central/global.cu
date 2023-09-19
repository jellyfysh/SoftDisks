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
#include "Soft.h"
__global__ void random_generate(curandStateMtgp32 (*state)[NCURAND]) {
  unsigned int n_parameter_set = (blockIdx.x + blockIdx.y * (int)GRIDL) % 200;
  unsigned int n_seed =
      (int)((float)(blockIdx.x + blockIdx.y * (int)GRIDL) / 200.0);

  float p = curand_uniform(&state[n_seed][n_parameter_set]);
  float q = curand_uniform(&state[n_seed][n_parameter_set]);
  printf("%d %f %f\n", blockIdx.y * 8 + blockIdx.x, p, q);
}

__global__ void JUST_MOVE_f(float *rrx1, float *rry1, short *nnparticle1,
                            float *rrx2, float *rry2, short *nnparticle2) {
  unsigned int index =
      ((blockIdx.x * MBLOCKL + threadIdx.x) % 2 == 0
           ? (blockIdx.x * MBLOCKL + threadIdx.x) / 2
           : (blockIdx.x * MBLOCKL + threadIdx.x + Nblock) / 2);
  unsigned int indey = blockIdx.y * MBLOCKL + threadIdx.y;
  nnparticle2[indey * Nblock + index] = nnparticle1[indey * Nblock + index];

  if (nnparticle1[indey * Nblock + index] >= nmax) {
    printf("Error on (%d, %d), %d\n", index, indey,
           nnparticle1[indey * Nblock + index]);
    assert(nnparticle1[indey * Nblock + index] < nmax);
  }

  for (int i = 0; i < nnparticle1[indey * Nblock + index]; i++) {
    rrx2[(i * Nblock + indey) * Nblock + index] =
        rrx1[(i * Nblock + indey) * Nblock + index];
    rry2[(i * Nblock + indey) * Nblock + index] =
        rry1[(i * Nblock + indey) * Nblock + index];
  }

  for (int i = nnparticle1[indey * Nblock + index]; i < nmax; i++) {
    rrx2[(i * Nblock + indey) * Nblock + index] = DUMMY_OFFSET;
    rry2[(i * Nblock + indey) * Nblock + index] = DUMMY_OFFSET;
  }
}
