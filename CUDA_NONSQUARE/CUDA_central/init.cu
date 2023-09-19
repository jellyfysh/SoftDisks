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
#include <curand_mtgp32_host.h>
#include <curand_mtgp32dc_p_11213.h>

void Set_Mersenne_Twister_GPU(int input){
  unsigned long int seed = 1000;
  int n_grid = (int)GRIDL * (int)GRIDL;
  int n_seed = (int)((double)n_grid / 200.0) + 1;

  cudaMalloc((void**)&devMTGPStates, NCURAND * n_seed * sizeof(curandStateMtgp32));
  cudaMalloc((void**)&devKernelParams, sizeof(mtgp32_kernel_params));
  curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, devKernelParams);
  //This is to let each block have a different seed for Mersenne Twister
  for(int i = 0; i < n_seed; i++)
    curandMakeMTGP32KernelState(devMTGPStates[i], mtgp32dc_params_fast_11213, devKernelParams,
				NCURAND, seed * (input + 1) * (i + 1));
}


void Set_init_conf(void){
  
  for(int j = 0; j < Nblock * Nblock; j++) nparticle[j] = 0;
  for(int j = 0; j < nmax * Nblock * Nblock; j++) rx[j] = DUMMY_OFFSET , ry[j] = DUMMY_OFFSET;


  if(reload() == 0)
    random_config();

  check_particle_number();

}


void random_config(void){
  
#ifdef RAND
  
  // Put particles on a triangular lattice
  int count = 0;
  int comptrNx = Nrow;
  int comptrNy = Nrow;
  
  float comptrspx = Lbox / (float)comptrNx;
  for(int y = 0; y < comptrNy; y++){
    for(int x = 0; x < comptrNx; x++){
      if(count < NNN){
        float px = 0.1 + comptrspx * ((float)x + (float)(y % 2) * 0.5);
        float py = comptrspx * y * sqrt(3.0) * 0.5;
        while(px < 0) px += Lbox;
      	while(py < 0) py += Lboy;
        while(px > Lbox) px -= Lbox;
        while(py > Lboy) py -= Lboy;
        int index = (int)(px / Lblockx);
        int indey = (int)(py / Lblocky);
	
        rx(index, indey, nparticle(index, indey)) = px - (float)index * Lblockx;
        ry(index, indey, nparticle(index, indey)) = py - (float)indey * Lblocky;

        nparticle(index, indey) += 1;
      }
      count++;
    }
  }



  int sum = 0;
  for(int index = 0; index < Nblock; index++)
    for(int indey = 0; indey < Nblock; indey++){
      sum += nparticle(index, indey);
    }

  cerr << "Sum is " << sum << endl;
  // Then simulate at high temperature to make the configuration random
  for(int i = 0; i < 10; i++) simulate(BETA * 0.1);

#endif

}
