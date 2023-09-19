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

__global__ void shift_checkerboard(float *rrx, float *rry, short *nnparticle,
                                   float rshift, int direction,
                                   double LLblock) {

  __shared__ float temp_rx[SHIFT_BLOCKL * SHIFT_BLOCKL * nmax],
      temp_ry[SHIFT_BLOCKL * SHIFT_BLOCKL * nmax];
  __shared__ unsigned int shifted_nparticle[SHIFT_BLOCKL * SHIFT_BLOCKL];

  short a = (direction % 2 == 0 ? (direction < 2 ? 1 : -1) : 0);
  short b = (direction % 2 == 1 ? (direction < 2 ? 1 : -1) : 0);

  for (short color = 0; color < 4; color++) {
    unsigned int xxx =
        (color % 2 == 0 ? SHIFT_BLOCKL * blockIdx.x + 2 * threadIdx.x
                        : SHIFT_BLOCKL * blockIdx.x + 2 * threadIdx.x + 1);
    unsigned int yyy =
        (color < 2 ? SHIFT_BLOCKL * blockIdx.y + 2 * threadIdx.y
                   : SHIFT_BLOCKL * blockIdx.y + 2 * threadIdx.y + 1);
    unsigned int index =
        (2 * threadIdx.y + (color < 2 ? 0 : 1)) * SHIFT_BLOCKL +
        (2 * threadIdx.x + (color % 2 == 0 ? 0 : 1));
    shifted_nparticle[index] = 0;

    unsigned int location =
        yyy * Nblock + (xxx % 2 == 0 ? xxx / 2 : (xxx + Nblock) / 2);

    for (int k = 0; k < tex1Dfetch(read_nparticle, location); k++) {
      float ppx = tex1Dfetch(read_rx, k * Nblock * Nblock + location) +
                  rshift * (float)a;
      float ppy = tex1Dfetch(read_ry, k * Nblock * Nblock + location) +
                  rshift * (float)b;

      short vecx = (direction % 2 == 0 ? (ppx > 0 && ppx < LLblock ? 0 : 1)
                                       : ((ppy > 0 && ppy < LLblock ? 0 : 1)));

      if (vecx == 0) {
        unsigned int j = index * nmax + shifted_nparticle[index];
        temp_rx[j] = ppx;
        temp_ry[j] = ppy;
        shifted_nparticle[index]++;
      }
    }

    location = ((yyy - b + Nblock) % Nblock) * Nblock +
               (((xxx - a + Nblock) % Nblock) % 2 == 0
                    ? ((xxx - a + Nblock) % Nblock) / 2
                    : ((xxx - a + Nblock) % Nblock + Nblock) / 2);

    for (int k = 0; k < tex1Dfetch(read_nparticle, location); k++) {

      float ppx = tex1Dfetch(read_rx, k * Nblock * Nblock + location) +
                  rshift * (float)a;
      float ppy = tex1Dfetch(read_ry, k * Nblock * Nblock + location) +
                  rshift * (float)b;

      short vecx = (direction % 2 == 0 ? (ppx > 0 && ppx < LLblock ? 0 : 1)
                                       : ((ppy > 0 && ppy < LLblock ? 0 : 1)));

      if (vecx == 1) {
        unsigned int j = index * nmax + shifted_nparticle[index];
        temp_rx[j] = ppx - LLblock * (float)a;
        temp_ry[j] = ppy - LLblock * (float)b;
        shifted_nparticle[index]++;
      }
    }

    location = yyy * Nblock + (xxx % 2 == 0 ? xxx / 2 : (xxx + Nblock) / 2);

    nnparticle[location] = shifted_nparticle[index];

    for (int k = 0; k < shifted_nparticle[index]; k++) {
      rrx[k * Nblock * Nblock + location] = temp_rx[index * nmax + k];
      rry[k * Nblock * Nblock + location] = temp_ry[index * nmax + k];
    }
  }
}
