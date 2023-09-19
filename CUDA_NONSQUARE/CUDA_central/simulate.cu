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

void simulate(float BBETA){
  dim3 grid(GRIDL, GRIDL);//Acticate GRIDL \times GRIDL blocks
  dim3 block(BLOCKL / 2, BLOCKL / 2);//Activate BLOCKL/2 \times BLOCKL/2 threads

  dim3 Sgrid(SHIFT_GRIDL, SHIFT_GRIDL);//Activate GRIDL \times GRIDL blocks
  dim3 Sblock(SHIFT_BLOCKL / 2, SHIFT_BLOCKL / 2);//Activate BLOCKL/2 \times BLOCKL/2 threads

  dim3 Mgrid(MGRIDL, MGRIDL);//Activate GRIDL \times GRIDL blocks
  dim3 Mblock(MBLOCKL, MBLOCKL);//Activate BLOCKL/2 \times BLOCKL/2 threads
  int color[4] = {0, 1, 2, 3};

  cudaDeviceSynchronize();
  // One loop below corresponds to 4x4x10 Monte Carlo sweeps
  for(int k = 0; k < 10; k++){
    short direction = (int)(4.0 * genrand(mt));
    float rshift = 0.5 * Lblocky * (float)genrand(mt);

    cudaBindTexture(0, read_rx, rx, cudaCreateChannelDesc<float>(), sizeof(float) * nmax * Nblock * Nblock);
    cudaBindTexture(0, read_ry, ry, cudaCreateChannelDesc<float>(), sizeof(float) * nmax * Nblock * Nblock);
    cudaBindTexture(0, read_nparticle, nparticle, cudaCreateChannelDesc<short>(), sizeof(short) * Nblock * Nblock);
    shift_checkerboard<<<Sgrid, Sblock>>>(devrx, devry, devnparticle, rshift, direction, Lblockx);
    cudaDeviceSynchronize();
    JUST_MOVE_f<<<Mgrid, Mblock>>>(devrx, devry, devnparticle, rx, ry, nparticle);
    cudaUnbindTexture(read_nparticle);
    cudaUnbindTexture(read_ry);
    cudaUnbindTexture(read_rx);

    rshiftx += (direction % 2 == 0 ? (direction < 2 ? 1.0 : -1.0) : 0.0) * rshift;
    rshifty += (direction % 2 == 1 ? (direction < 2 ? 1.0 : -1.0) : 0.0) * rshift;
    
    cudaDeviceSynchronize();

    for(int i = 0; i < 10; i++){
      int a = (int)(genrand(mt) * 4.0);
      int b = (a + (int)(genrand(mt) * 3.0)) % 4;
      int p = color[a];
      color[a] = color[b];
      color[b] = p;
    }
    cudaDeviceSynchronize();
    cudaBindTexture(0, read_rx, rx, cudaCreateChannelDesc<float>(), sizeof(float) * nmax * Nblock * Nblock);
    cudaBindTexture(0, read_ry, ry, cudaCreateChannelDesc<float>(), sizeof(float) * nmax * Nblock * Nblock);
    cudaBindTexture(0, read_nparticle, nparticle, cudaCreateChannelDesc<short>(), sizeof(short) * Nblock * Nblock);
    // One loop below corresponds to 4x4 Monte Carlo sweeps
    for(int m = 0; m < 4; m++){
      // One call of update_Metropolis corresponds to 4 Monte Carlo sweeps
      update_Metropolis<<<grid, block>>>(rx, ry, color[m], rshiftx, rshifty, Lblockx, Lblocky, Lbox, BBETA, devMTGPStates);
    }

    cudaUnbindTexture(read_nparticle);
    cudaUnbindTexture(read_rx);
    cudaUnbindTexture(read_ry);
    cudaDeviceSynchronize();
    
  }
  cudaDeviceSynchronize();
}
