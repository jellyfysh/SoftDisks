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
#include <fstream>

#include <highfive/H5DataSpace.hpp>
#include <highfive/H5Easy.hpp>
#include <highfive/H5File.hpp>

#include <boost/asio/ip/host_name.hpp>
#include <pwd.h>
#include <sys/types.h>
#include <unistd.h>

#include "Soft.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace std;
using namespace std;
using namespace HighFive;
using namespace H5Easy;

int reload(void) {

  ifstream reload("start.h5");
  if (reload.good()) {
    cout << "Found" << endl;
    File fd("start.h5", File::ReadOnly);
    float l_beta = load<float>(fd, "/parameters/BETA");
    cout << "beta " << l_beta << endl;
    int l_rpotential = load<int>(fd, "/parameters/rpotential");
    cout << "potential " << l_rpotential << endl;
    float l_density = load<float>(fd, "/parameters/density");
    cout << "density " << l_density << endl;
    float l_cutoff = load<float>(fd, "/parameters/cutoff");
    cout << "cutoff" << l_cutoff << endl;

    int l_nrow = load<int>(fd, "/parameters/Nrow");
    cout << "nrow " << l_nrow << endl;

    auto config = load<vector<vector<float>>>(fd, "/config");
    auto index = load<vector<vector<int>>>(fd, "/index");

    cout << "Start of new configuration" << endl;
    int test = 0;
    for (int i = 0; i < NNN; i++) {

      rx(index[i][0], index[i][1], nparticle(index[i][0], index[i][1])) =
          config[i][0];
      ry(index[i][0], index[i][1], nparticle(index[i][0], index[i][1])) =
          config[i][1];

      if (i < 10)
        cout << rx(index[i][0], index[i][1],
                   nparticle(index[i][0], index[i][1]))
             << " "
             << ry(index[i][0], index[i][1],
                   nparticle(index[i][0], index[i][1]))
             << endl;

      nparticle(index[i][0], index[i][1]) += 1;
      // test++;
    }
    for (int i = 0; i < Nblock * Nblock; i++)
      test += nparticle[i];
    cerr << "Number is " << test << endl;

    if (fabs(l_density - density) > 1.e-5) {
      cerr << "Density incorrect in reload" << endl;
      exit(1);
    }

    if (fabs(l_cutoff - cutoff) > 1.e-5) {
      cerr << "Density incorrect in reload" << endl;
    }

    if (fabs(l_cutoff - cutoff) > 1.e-5) {
      cerr << "Density incorrect in reload" << endl;
    }

    return 1;
  }
  return 0;
}
