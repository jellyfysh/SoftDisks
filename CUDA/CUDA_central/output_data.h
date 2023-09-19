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
#ifndef _OUTPUT_DATA
#define _OUTPUT_DATA

#include <algorithm>
#include <assert.h>
#include <iostream>
#include <math.h>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

using namespace std;

void create(int, float, float, float, string &);
void dump_config(int, string &, float, float, float);
void write_ep(vector<float> &, vector<float> &, vector<float> &, string &, int);
void write_time(string &, double, double);

extern int freq_output;
extern int freq_energy;

#endif
