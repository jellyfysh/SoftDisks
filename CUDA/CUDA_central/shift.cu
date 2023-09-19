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

void shift_checkerboard_CPU(float offsetx, float offsety) {
  int temp_nparticle[Nblock * Nblock] = {0};
  float temp_rx[Nblock * Nblock * nmax], temp_ry[Nblock * Nblock * nmax];

  for (int x = 0; x < Nblock; x++) {
    for (int y = 0; y < Nblock; y++) {
      for (int k = 0; k < nparticle(x, y); k++) {

        int vecx = (int)((rx(x, y, k) + offsetx) / Lblock + 1.0) - 1;
        int vecy = (int)((ry(x, y, k) + offsety) / Lblock + 1.0) - 1;

        temp_rx((x + vecx + Nblock) % Nblock, (y + vecy + Nblock) % Nblock,
                temp_nparticle((x + vecx + Nblock) % Nblock,
                               (y + vecy + Nblock) % Nblock)) =
            rx(x, y, k) + offsetx - vecx * Lblock;
        temp_ry((x + vecx + Nblock) % Nblock, (y + vecy + Nblock) % Nblock,
                temp_nparticle((x + vecx + Nblock) % Nblock,
                               (y + vecy + Nblock) % Nblock)) =
            ry(x, y, k) + offsety - vecy * Lblock;
        temp_nparticle((x + vecx + Nblock) % Nblock,
                       (y + vecy + Nblock) % Nblock)++;
      }
    }
  }

  for (int x = 0; x < Nblock * Nblock; x++) {
    if (temp_nparticle[x] >= nmax) {
      cerr << "Error" << x << " " << temp_nparticle[x] << endl;
      assert(temp_nparticle[x] < nmax);
    }
    nparticle[x] = temp_nparticle[x];
  }

  for (int i = 0; i < Nblock * Nblock * nmax; i++) {
    rx[i] = temp_rx[i];
    ry[i] = temp_ry[i];
  }
}