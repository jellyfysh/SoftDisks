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
#include <fstream>
void measure_e_p(double &outpute, double &outputp, double &outputhv) {
  outputp = 0;
  outpute = 0;
  outputhv = 0;

  float sq_cutoff = cutoff * cutoff;
  float shift_energy = powq(sq_cutoff, -0.5 * rpotential);
  for (int i = 0; i < Nblock; i++) {
    for (int j = 0; j < Nblock; j++) {
      for (int k = 0; k < nparticle(i, j); k++) {

        for (int x = -1; x <= 1; x++) {
          for (int y = -1; y <= 1; y++) {
            for (int m = 0; m < nparticle((i + x + Nblock) % Nblock,
                                          (j + y + Nblock) % Nblock);
                 m++) {
              float x0 = rx(i, j, k);
              float y0 = ry(i, j, k);
              float x1 =
                  rx((i + x + Nblock) % Nblock, (j + y + Nblock) % Nblock, m) +
                  (float)x * Lblock;
              float y1 =
                  ry((i + x + Nblock) % Nblock, (j + y + Nblock) % Nblock, m) +
                  (float)y * Lblock;

              float dist_x = (x0 - x1);
              float dist_y = (y0 - y1);
              float sq_dist = (dist_x * dist_x + dist_y * dist_y);
              if (sq_dist > 0 && sq_dist < sq_cutoff) {
                float rpowe = powq(sq_dist, -0.5 * rpotential);
                float rpowp = powp(sq_dist, -0.5 * rpotential);
                outputp += 0.5 * 0.5 * rpowp;
                outpute += 0.5 * (rpowe - shift_energy);
                outputhv += 0.5 * (powh(sq_dist, -0.5 * rpotential) - rpowp);
              }
            }
          }
        }
      }
    }
  }
  outputp = (density * outputp / (double)NNN) + density / BETA;
  outpute /= NNN;
  outputhv /= 4.0 * (double)NNN / density;
}

void check_particle_number(void) {

  int sum_particle = 0;
  for (int i = 0; i < Nblock * Nblock; i++) {
    if (nparticle[i] >= nmax) {
      cerr << "ERROR " << nparticle[i] << endl;
      assert(nparticle[i] < nmax);
    }
    sum_particle += nparticle[i];
  }

  if (sum_particle != NNN)
    cerr << "Error " << sum_particle << " != " << NNN << endl,
        assert(sum_particle == NNN);
}
