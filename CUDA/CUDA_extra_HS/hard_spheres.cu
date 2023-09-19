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
#include "../CUDA_central/Soft.h"
#include "../CUDA_central/fastpow.h"

curandStateMtgp32 (*devMTGPStates)[NCURAND];
mtgp32_kernel_params *devKernelParams;

// Mersenne Twister on CPU
random_device rd;
mt19937 mt(111); // Seed should be fixed
uniform_real_distribution<double> genrand(0.0, 1.0);

__global__ void update_Metropolis(float *rrx, float *rry, unsigned int color,
                                  float rshiftx, float rshifty, float LLblock,
                                  double LLbox, float Beta,
                                  curandStateMtgp32 (*state)[NCURAND]) {

  __shared__ float temp_rx[nmax * BLOCKL / 2 * BLOCKL / 2],
      temp_ry[nmax * BLOCKL / 2 * BLOCKL / 2];

  float blockL = LLblock;
  // double boxL = LLbox;
  unsigned int n_parameter_set = (blockIdx.x + blockIdx.y * (int)GRIDL) % 200;
  unsigned int n_seed =
      (unsigned int)((float)(blockIdx.x + blockIdx.y * (int)GRIDL) / 200.0);

  unsigned int xxx =
      (color % 2 == 0 ? BLOCKL * blockIdx.x + 2 * threadIdx.x
                      : BLOCKL * blockIdx.x + 2 * threadIdx.x + 1);
  unsigned int yyy = (color < 2 ? BLOCKL * blockIdx.y + 2 * threadIdx.y
                                : BLOCKL * blockIdx.y + 2 * threadIdx.y + 1);
  unsigned int position =
      yyy * Nblock + (xxx % 2 == 0 ? xxx / 2 : (xxx + Nblock) / 2);
  unsigned int index = threadIdx.y * BLOCKL / 2 + threadIdx.x;
  unsigned int bNNN = Nblock * Nblock;
  const unsigned int temp_nparticle = tex1Dfetch(read_nparticle, position);

  for (short i = 0; i < nmax; i++) {
    temp_rx[(nmax * index + i)] = tex1Dfetch(read_rx, i * bNNN + position); //
    temp_ry[(nmax * index + i)] = tex1Dfetch(read_ry, i * bNNN + position);
  }

  unsigned int order[nloop];
  for (unsigned int j = 0; j < nloop; j++) {
    order[j] = (int)((float)temp_nparticle *
                     (curand_uniform(&state[n_seed][n_parameter_set])));
  }

  const float sq_cutoff = cutoff * cutoff;
  const float shift_cutoff = powq(sq_cutoff, -rpotential * 0.5);

  for (int k = 0; k < nloop; k++) {
    unsigned int pindex = order[k];

    float prex = temp_rx[(nmax * index + pindex)];
    float prey = temp_ry[(nmax * index + pindex)];

    float nexx =
        prex + 0.32 * (curand_uniform(&state[n_seed][n_parameter_set]) -
                       0.5); // This should be optimized
    float nexy =
        prey + 0.32 * (curand_uniform(&state[n_seed][n_parameter_set]) - 0.5);
    unsigned short judge = 1;

    //=== 1st block ===//
    for (unsigned int j = 0; j < temp_nparticle; j++) {
      if (j != pindex) {

        float nexdist = (nexx - temp_rx[(nmax * index + j)]) *
                            (nexx - temp_rx[(nmax * index + j)]) +
                        (nexy - temp_ry[(nmax * index + j)]) *
                            (nexy - temp_ry[(nmax * index + j)]);

        judge = (nexdist < 1.0 ? 0 : judge & 1);
      }
    }
    //=== 2nd block ===//

    short nn_block =
        tex1Dfetch(read_nparticle,
                   yyy * Nblock + (((xxx + 1) % Nblock) % 2 == 0
                                       ? ((xxx + 1) % Nblock) / 2
                                       : ((xxx + 1) % Nblock + Nblock) / 2));

    for (unsigned int j = 0; j < nn_block; j++) {
      float nexdist =
          (nexx -
           tex1Dfetch(read_rx, ((j * Nblock + yyy) * Nblock +
                                (((xxx + 1) % Nblock) % 2 == 0
                                     ? ((xxx + 1) % Nblock) / 2
                                     : ((xxx + 1) % Nblock + Nblock) / 2))) -
           blockL) *
              (nexx -
               tex1Dfetch(read_rx,
                          ((j * Nblock + yyy) * Nblock +
                           (((xxx + 1) % Nblock) % 2 == 0
                                ? ((xxx + 1) % Nblock) / 2
                                : ((xxx + 1) % Nblock + Nblock) / 2))) -
               blockL) +
          (nexy -
           tex1Dfetch(read_ry, (j * Nblock + yyy) * Nblock +
                                   (((xxx + 1) % Nblock) % 2 == 0
                                        ? ((xxx + 1) % Nblock) / 2
                                        : ((xxx + 1) % Nblock + Nblock) / 2))) *
              (nexy -
               tex1Dfetch(read_ry,
                          (j * Nblock + yyy) * Nblock +
                              (((xxx + 1) % Nblock) % 2 == 0
                                   ? ((xxx + 1) % Nblock) / 2
                                   : ((xxx + 1) % Nblock + Nblock) / 2)));

      judge = (nexdist < 1.0 ? 0 : judge & 1);
    }
    //=== 3rd block ===//

    nn_block = tex1Dfetch(
        read_nparticle,
        yyy * Nblock + (((xxx - 1 + Nblock) % Nblock) % 2 == 0
                            ? ((xxx - 1 + Nblock) % Nblock) / 2
                            : ((xxx - 1 + Nblock) % Nblock + Nblock) / 2));

    for (unsigned int j = 0; j < nn_block; j++) {
      float nexdist =
          (nexx -
           tex1Dfetch(read_rx,
                      (j * Nblock + yyy) * Nblock +
                          (((xxx - 1 + Nblock) % Nblock) % 2 == 0
                               ? ((xxx - 1 + Nblock) % Nblock) / 2
                               : ((xxx - 1 + Nblock) % Nblock + Nblock) / 2)) +
           blockL) *
              (nexx -
               tex1Dfetch(
                   read_rx,
                   (j * Nblock + yyy) * Nblock +
                       (((xxx - 1 + Nblock) % Nblock) % 2 == 0
                            ? ((xxx - 1 + Nblock) % Nblock) / 2
                            : ((xxx - 1 + Nblock) % Nblock + Nblock) / 2)) +
               blockL) +
          (nexy -
           tex1Dfetch(read_ry,
                      (j * Nblock + yyy) * Nblock +
                          (((xxx - 1 + Nblock) % Nblock) % 2 == 0
                               ? ((xxx - 1 + Nblock) % Nblock) / 2
                               : ((xxx - 1 + Nblock) % Nblock + Nblock) / 2))) *
              (nexy -
               tex1Dfetch(
                   read_ry,
                   (j * Nblock + yyy) * Nblock +
                       (((xxx - 1 + Nblock) % Nblock) % 2 == 0
                            ? ((xxx - 1 + Nblock) % Nblock) / 2
                            : ((xxx - 1 + Nblock) % Nblock + Nblock) / 2)));

      judge = (nexdist < 1.0 ? 0 : judge & 1);
    }
    //=== 4th block ===//

    nn_block = tex1Dfetch(read_nparticle,
                          ((yyy + 1) % Nblock) * Nblock +
                              (xxx % 2 == 0 ? xxx / 2 : (xxx + Nblock) / 2));

    for (unsigned int j = 0; j < nn_block; j++) {
      float nexdist =
          (nexx -
           tex1Dfetch(read_rx,
                      (j * Nblock + ((yyy + 1) % Nblock)) * Nblock +
                          (xxx % 2 == 0 ? xxx / 2 : (xxx + Nblock) / 2))) *
              (nexx -
               tex1Dfetch(read_rx,
                          (j * Nblock + ((yyy + 1) % Nblock)) * Nblock +
                              (xxx % 2 == 0 ? xxx / 2 : (xxx + Nblock) / 2))) +
          (nexy -
           tex1Dfetch(read_ry,
                      (j * Nblock + ((yyy + 1) % Nblock)) * Nblock +
                          (xxx % 2 == 0 ? xxx / 2 : (xxx + Nblock) / 2)) -
           blockL) *
              (nexy -
               tex1Dfetch(read_ry,
                          (j * Nblock + ((yyy + 1) % Nblock)) * Nblock +
                              (xxx % 2 == 0 ? xxx / 2 : (xxx + Nblock) / 2)) -
               blockL);

      judge = (nexdist < 1.0 ? 0 : judge & 1);
    }
    //=== 5th block ===//

    nn_block = tex1Dfetch(read_nparticle,
                          ((yyy - 1 + Nblock) % Nblock) * Nblock +
                              (xxx % 2 == 0 ? xxx / 2 : (xxx + Nblock) / 2));

    for (unsigned int j = 0; j < nn_block; j++) {
      float nexdist =
          (nexx -
           tex1Dfetch(read_rx,
                      (j * Nblock + ((yyy - 1 + Nblock) % Nblock)) * Nblock +
                          (xxx % 2 == 0 ? xxx / 2 : (xxx + Nblock) / 2))) *
              (nexx -
               tex1Dfetch(read_rx,
                          (j * Nblock + ((yyy - 1 + Nblock) % Nblock)) *
                                  Nblock +
                              (xxx % 2 == 0 ? xxx / 2 : (xxx + Nblock) / 2))) +
          (nexy -
           tex1Dfetch(read_ry,
                      (j * Nblock + ((yyy - 1 + Nblock) % Nblock)) * Nblock +
                          (xxx % 2 == 0 ? xxx / 2 : (xxx + Nblock) / 2)) +
           blockL) *
              (nexy -
               tex1Dfetch(read_ry,
                          (j * Nblock + ((yyy - 1 + Nblock) % Nblock)) *
                                  Nblock +
                              (xxx % 2 == 0 ? xxx / 2 : (xxx + Nblock) / 2)) +
               blockL);

      judge = (nexdist < 1.0 ? 0 : judge & 1);
    }
    //=== 6th block ===//

    nn_block = tex1Dfetch(read_nparticle,
                          ((yyy + 1) % Nblock) * Nblock +
                              (((xxx + 1) % Nblock) % 2 == 0
                                   ? ((xxx + 1) % Nblock) / 2
                                   : ((xxx + 1) % Nblock + Nblock) / 2));

    for (unsigned int j = 0; j < nn_block; j++) {
      float nexdist =
          (nexx -
           tex1Dfetch(read_rx, (j * Nblock + ((yyy + 1) % Nblock)) * Nblock +
                                   (((xxx + 1) % Nblock) % 2 == 0
                                        ? ((xxx + 1) % Nblock) / 2
                                        : ((xxx + 1) % Nblock + Nblock) / 2)) -
           blockL) *
              (nexx -
               tex1Dfetch(read_rx,
                          (j * Nblock + ((yyy + 1) % Nblock)) * Nblock +
                              (((xxx + 1) % Nblock) % 2 == 0
                                   ? ((xxx + 1) % Nblock) / 2
                                   : ((xxx + 1) % Nblock + Nblock) / 2)) -
               blockL) +
          (nexy -
           tex1Dfetch(read_ry, (j * Nblock + ((yyy + 1) % Nblock)) * Nblock +
                                   (((xxx + 1) % Nblock) % 2 == 0
                                        ? ((xxx + 1) % Nblock) / 2
                                        : ((xxx + 1) % Nblock + Nblock) / 2)) -
           blockL) *
              (nexy -
               tex1Dfetch(read_ry,
                          (j * Nblock + ((yyy + 1) % Nblock)) * Nblock +
                              (((xxx + 1) % Nblock) % 2 == 0
                                   ? ((xxx + 1) % Nblock) / 2
                                   : ((xxx + 1) % Nblock + Nblock) / 2)) -
               blockL);

      judge = (nexdist < 1.0 ? 0 : judge & 1);
    }
    //=== 7th block ===//

    nn_block = tex1Dfetch(
        read_nparticle, ((yyy - 1 + Nblock) % Nblock) * Nblock +
                            (((xxx - 1 + Nblock) % Nblock) % 2 == 0
                                 ? ((xxx - 1 + Nblock) % Nblock) / 2
                                 : ((xxx - 1 + Nblock) % Nblock + Nblock) / 2));

    for (unsigned int j = 0; j < nn_block; j++) {
      float nexdist =
          (nexx -
           tex1Dfetch(read_rx,
                      (j * Nblock + ((yyy - 1 + Nblock) % Nblock)) * Nblock +
                          (((xxx - 1 + Nblock) % Nblock) % 2 == 0
                               ? ((xxx - 1 + Nblock) % Nblock) / 2
                               : ((xxx - 1 + Nblock) % Nblock + Nblock) / 2)) +
           blockL) *
              (nexx -
               tex1Dfetch(
                   read_rx,
                   (j * Nblock + ((yyy - 1 + Nblock) % Nblock)) * Nblock +
                       (((xxx - 1 + Nblock) % Nblock) % 2 == 0
                            ? ((xxx - 1 + Nblock) % Nblock) / 2
                            : ((xxx - 1 + Nblock) % Nblock + Nblock) / 2)) +
               blockL) +
          (nexy -
           tex1Dfetch(read_ry,
                      (j * Nblock + ((yyy - 1 + Nblock) % Nblock)) * Nblock +
                          (((xxx - 1 + Nblock) % Nblock) % 2 == 0
                               ? ((xxx - 1 + Nblock) % Nblock) / 2
                               : ((xxx - 1 + Nblock) % Nblock + Nblock) / 2)) +
           blockL) *
              (nexy -
               tex1Dfetch(
                   read_ry,
                   (j * Nblock + ((yyy - 1 + Nblock) % Nblock)) * Nblock +
                       (((xxx - 1 + Nblock) % Nblock) % 2 == 0
                            ? ((xxx - 1 + Nblock) % Nblock) / 2
                            : ((xxx - 1 + Nblock) % Nblock + Nblock) / 2)) +
               blockL);

      judge = (nexdist < 1.0 ? 0 : judge & 1);
    }
    //=== 8th block ===//

    nn_block = tex1Dfetch(read_nparticle,
                          (((yyy - 1 + Nblock) % Nblock)) * Nblock +
                              (((xxx + 1) % Nblock) % 2 == 0
                                   ? ((xxx + 1) % Nblock) / 2
                                   : ((xxx + 1) % Nblock + Nblock) / 2));

    for (unsigned int j = 0; j < nn_block; j++) {
      float nexdist =
          (nexx -
           tex1Dfetch(read_rx,
                      (j * Nblock + ((yyy - 1 + Nblock) % Nblock)) * Nblock +
                          (((xxx + 1) % Nblock) % 2 == 0
                               ? ((xxx + 1) % Nblock) / 2
                               : ((xxx + 1) % Nblock + Nblock) / 2)) -
           blockL) *
              (nexx -
               tex1Dfetch(read_rx,
                          (j * Nblock + ((yyy - 1 + Nblock) % Nblock)) *
                                  Nblock +
                              (((xxx + 1) % Nblock) % 2 == 0
                                   ? ((xxx + 1) % Nblock) / 2
                                   : ((xxx + 1) % Nblock + Nblock) / 2)) -
               blockL) +
          (nexy -
           tex1Dfetch(read_ry,
                      (j * Nblock + ((yyy - 1 + Nblock) % Nblock)) * Nblock +
                          (((xxx + 1) % Nblock) % 2 == 0
                               ? ((xxx + 1) % Nblock) / 2
                               : ((xxx + 1) % Nblock + Nblock) / 2)) +
           blockL) *
              (nexy -
               tex1Dfetch(read_ry,
                          (j * Nblock + ((yyy - 1 + Nblock) % Nblock)) *
                                  Nblock +
                              (((xxx + 1) % Nblock) % 2 == 0
                                   ? ((xxx + 1) % Nblock) / 2
                                   : ((xxx + 1) % Nblock + Nblock) / 2)) +
               blockL);

      judge = (nexdist < 1.0 ? 0 : judge & 1);
    }
    //=== 9th block ===//

    nn_block = tex1Dfetch(
        read_nparticle, (((yyy + 1) % Nblock)) * Nblock +
                            (((xxx - 1 + Nblock) % Nblock) % 2 == 0
                                 ? ((xxx - 1 + Nblock) % Nblock) / 2
                                 : ((xxx - 1 + Nblock) % Nblock + Nblock) / 2));

    for (unsigned int j = 0; j < nn_block; j++) {

      float nexdist =
          (nexx -
           tex1Dfetch(read_rx,
                      (j * Nblock + ((yyy + 1) % Nblock)) * Nblock +
                          (((xxx - 1 + Nblock) % Nblock) % 2 == 0
                               ? ((xxx - 1 + Nblock) % Nblock) / 2
                               : ((xxx - 1 + Nblock) % Nblock + Nblock) / 2)) +
           blockL) *
              (nexx -
               tex1Dfetch(
                   read_rx,
                   (j * Nblock + ((yyy + 1) % Nblock)) * Nblock +
                       (((xxx - 1 + Nblock) % Nblock) % 2 == 0
                            ? ((xxx - 1 + Nblock) % Nblock) / 2
                            : ((xxx - 1 + Nblock) % Nblock + Nblock) / 2)) +
               blockL) +
          (nexy -
           tex1Dfetch(read_ry,
                      (j * Nblock + ((yyy + 1) % Nblock)) * Nblock +
                          (((xxx - 1 + Nblock) % Nblock) % 2 == 0
                               ? ((xxx - 1 + Nblock) % Nblock) / 2
                               : ((xxx - 1 + Nblock) % Nblock + Nblock) / 2)) -
           blockL) *
              (nexy -
               tex1Dfetch(
                   read_ry,
                   (j * Nblock + ((yyy + 1) % Nblock)) * Nblock +
                       (((xxx - 1 + Nblock) % Nblock) % 2 == 0
                            ? ((xxx - 1 + Nblock) % Nblock) / 2
                            : ((xxx - 1 + Nblock) % Nblock + Nblock) / 2)) -
               blockL);

      judge = (nexdist < 1.0 ? 0 : judge & 1);
    }

    if (judge == 1 && nexx > 0.0 && nexy > 0.0 && nexx < blockL &&
        nexy < blockL) {
      temp_rx[(index * nmax + pindex)] = nexx;
      temp_ry[(index * nmax + pindex)] = nexy;
    }
  }
  __syncthreads();

  for (int j = 0; j < temp_nparticle; j++) {
    rrx[j * bNNN + position] = temp_rx[nmax * index + j];
    rry[j * bNNN + position] = temp_ry[nmax * index + j];
  }
}
