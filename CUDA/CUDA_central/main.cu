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
#include "output_data.h"
#include "timer.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
int freq_output = -1;
int freq_energy = 50;

float density;
double Lbox;
double Lblock;
float *rx, *ry;
short *nparticle;
float *devrx, *devry;
short *devnparticle;
float rshiftx = 0, rshifty = 0;
float BETA;

// Texture memory
texture<float, 1, cudaReadModeElementType> read_rx;
texture<float, 1, cudaReadModeElementType> read_ry;
texture<short, 1, cudaReadModeElementType> read_nparticle;

int main(int argc, char **argv) {
  Timer mytime;
#ifdef INPUT_DENSITY

  if (argc != 3) {
    cerr << "Input density and BETA" << endl;
    assert(argc == 3);
  } else {
    density = atof(argv[1]);
    BETA = atof(argv[2]);
  }

  Lbox = sqrt((double)NNN / (double)density);
  Lblock = (float)(Lbox / (double)Nblock);
  if (Lblock < cutoff) {
    cerr << "Error, Lblock " << Lblock << " must be larger than cutoff "
         << cutoff << endl;
    assert(Lblock > cutoff);
  }
#endif

  cerr.precision(12);
  cerr << " N = " << NNN << ", density = " << density << ", BETA = " << BETA
       << ", Lbox = " << Lbox << ", Lblock = " << Lblock << endl;
  cerr << "Lbox\t" << Lbox << endl;

  int num_gpus, gpu_id;
  cudaGetDeviceCount(&num_gpus);
  cudaGetDevice(&gpu_id);

  if (1) {
    cout << "Nrow\t" << Nrow << endl;
    cout << "NNN\t" << NNN << endl;
    cout << "rpotential\t" << rpotential << endl;
    cout << "MaxMCS\t" << MaxMCS << endl;
    cout << "cutoff\t" << cutoff << endl;
    cout << "Nblock\t" << Nblock << endl;
    cout << "nmax\t" << nmax << endl;
    cout << "BLOCKL\t" << BLOCKL << endl;
    cout << "GRIDL\t" << GRIDL << endl;
    cout << "BLOCKS\t" << BLOCKS << endl;
    cout << "THREADS\t" << THREADS << endl;
    cout << "SHIFT_BLOCKL\t" << endl;
    cout << "SHIFT_GRIDL\t" << SHIFT_GRIDL << endl;
    cout << "MBLOCKL\t" << MBLOCKL << endl;
    cout << "MGRIDL\t" << MGRIDL << endl;
    cout << "NCURAND\t" << NCURAND << endl << endl;
  }

  { // Allocate memory on gpu
    cudaMallocManaged(&rx, Nblock * Nblock * nmax * sizeof(float));
    cudaMallocManaged(&ry, Nblock * Nblock * nmax * sizeof(float));
    cudaMallocManaged(&nparticle, Nblock * Nblock * sizeof(short));
    cudaMallocManaged(&devrx, Nblock * Nblock * nmax * sizeof(float));
    cudaMallocManaged(&devry, Nblock * Nblock * nmax * sizeof(float));
    cudaMallocManaged(&devnparticle, Nblock * Nblock * sizeof(short));
  }

  int seed = (int)(density * 100 + NNN) + time(0);
  Set_Mersenne_Twister_GPU(seed);
  cudaDeviceSynchronize();

  cout.precision(7);
  cout.setf(ios::scientific);

  freq_output = (MaxMCS < N_config ? MaxMCS : (int)(MaxMCS / N_config));
  if (rpotential < 0)
    freq_energy = 50;

  Set_init_conf();
  unsigned int iter = 0;
  ostringstream ss;
  ss << setw(3) << setfill('0') << iter;
  string output_file = "data-" + ss.str() + ".h5";
  cerr << "Trial " << output_file << endl;
  ifstream output_name;
  output_name.open(output_file);
  while (output_name.is_open()) {
    output_name.close();
    iter++;
    ss.str("");
    ss.clear();
    ss << setw(3) << setfill('0') << iter;
    output_file = "data-" + ss.str() + ".h5";
    cerr << "Trial " << output_file << endl;
    output_name.open(output_file);
  }

  double energy = 0, pressure = 0, hypervirial = 0;

  int count_list = 0, size_list = 1000;
  vector<float> list_pressure(size_list), list_energy(size_list),
      list_hypervirial(size_list);

#if defined(SOFTCUDA)
  measure_e_p(energy, pressure, hypervirial);
  create(seed, (float)energy, (float)pressure, (float)hypervirial, output_file);
#endif
#if defined(HARDCUDA)
  double pressure_y = 0;
  vector<float> list_pressure_y(size_list);
  measure_pressure(pressure, pressure_y);
  create(seed, 0, (float)pressure, (float)pressure_y, output_file);
#endif
  write_time(output_file, 0, 0);

  int count_output = 0;

  for (int MCS = 0; MCS < MaxMCS; MCS++) {

    simulate(BETA);

    if (MCS % freq_energy == 0) {
#if defined(SOFTCUDA)
      measure_e_p(energy, pressure, hypervirial);
      list_energy[count_list] = (float)energy;
      list_pressure[count_list] = (float)pressure;
      list_hypervirial[count_list] = (float)hypervirial;
      count_list++;
      if (count_list == size_list) {
        write_ep(list_energy, list_pressure, list_hypervirial, output_file,
                 count_list);
        count_list = 0;
      }
#elif defined(HARDCUDA)
      measure_pressure(pressure, pressure_y);
      list_pressure[count_list] = (float)pressure;
      list_pressure_y[count_list] = (float)pressure_y;
      count_list++;
      if (count_list == size_list) {
        write_ep(list_energy, list_pressure, list_pressure_y, output_file,
                 count_list);
        count_list = 0;
      }
#endif
    }

    if (MCS % freq_output == 0) {
      check_particle_number();
      dump_config(count_output, output_file, Lbox, rshiftx, rshifty);

      count_output++;
    }
  }

  check_particle_number();
  dump_config(count_output, output_file, Lbox, rshiftx, rshifty);
#if defined(SOFTCUDA)
  write_ep(list_energy, list_pressure, list_hypervirial, output_file,
           count_list);
#elif defined(HARDCUDA)
  write_ep(list_energy, list_pressure, list_pressure_y, output_file,
           count_list);
#endif
  write_time(output_file, mytime.cpu(), mytime.clock());

  mytime.stats();
  cout << "Run finished" << endl;
  cerr << "Run finished" << endl;

  return 0;
}
