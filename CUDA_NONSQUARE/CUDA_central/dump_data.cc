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
#include <fstream>

#include <highfive/H5DataSpace.hpp>
#include <highfive/H5Easy.hpp>
#include <highfive/H5File.hpp>

#include <boost/asio/ip/host_name.hpp>
#include <pwd.h>
#include <sys/types.h>
#include <unistd.h>

#include "../version.h"

using namespace HighFive;
using namespace H5Easy;

void create(int seed, float init_e, float init_p, float init_hv,
            string &write_file) {

  File fd(write_file, File::ReadWrite | File::Create | File::Truncate);
  string version_string = GIT_COMMIT;
  dump(fd, "parameters/git_version", version_string);
  dump(fd, "parameters/density", density);
  dump(fd, "parameters/BETA", BETA);
  dump(fd, "parameters/Nrow", Nrow);
  dump(fd, "parameters/NNN", NNN);
  dump(fd, "parameters/rpotential", rpotential);
  dump(fd, "parameters/MaxMCS", MaxMCS);
  dump(fd, "parameters/cutoff", cutoff);
  dump(fd, "parameters/gpu/Nblock", Nblock);
  dump(fd, "parameters/gpu/nmax", nmax);
  dump(fd, "parameters/gpu/BLOCKL", BLOCKL);
  dump(fd, "parameters/gpu/GRIDL", GRIDL);
  dump(fd, "parameters/gpu/THREADS", THREADS);
  dump(fd, "parameters/gpu/SHIFT_BLOCKL", SHIFT_BLOCKL);
  dump(fd, "parameters/gpu/SHIFT_GRIDL", SHIFT_GRIDL);
  dump(fd, "parameters/gpu/MBLOCKL", MBLOCKL);
  dump(fd, "parameters/gpu/MGRIDL", MGRIDL);
  dump(fd, "parameters/gpu/NCURAND", NCURAND);
  dump(fd, "parameters/seed", seed);
  dump(fd, "parameters/freq_output", freq_output);
  dump(fd, "parameters/freq_energy", freq_energy);

  string hostname = boost::asio::ip::host_name();
  dump(fd, "parameters/hostname", hostname);
  uid_t uid = geteuid();
  struct passwd *pw = getpwuid(uid);
  string username = string(pw->pw_name);
  dump(fd, "parameters/username", username);

  DataSpace ds = DataSpace({1}, {DataSpace::UNLIMITED});
  DataSetCreateProps props;
  props.add(Chunking(vector<hsize_t>{1}));

  DataSet pset = fd.createDataSet("/pressure", ds, AtomicType<float>(), props);
  pset.select({0}, {1}).write(init_p);

  if (rpotential > 0) { // The following will be done when the interaction
                        // potential is WCA or IPL
    DataSet eset = fd.createDataSet("/energy", ds, AtomicType<float>(), props);
    DataSet hvset =
        fd.createDataSet("/hypervirial", ds, AtomicType<float>(), props);
    eset.select({0}, {1}).write(init_e);
    hvset.select({0}, {1}).write(init_hv);
  } else {
    DataSet pyset =
        fd.createDataSet("/pressurey", ds, AtomicType<float>(), props);
    pyset.select({0}, {1}).write(init_hv);
  }
}

void dump_config(int conf_number, string &write_file, float Lbox, float rshiftx,
                 float rshifty) {

  File fd(write_file, File::ReadWrite);
  double Lboy = sqrt(3.0) * 0.5 * Lbox;
  string ds_name =
      "/config-" + to_string(conf_number); // string with name of dataset

  vector<size_t> dims{NNN, 2};
  DataSet config = fd.createDataSet<double>(ds_name, DataSpace(dims));
  dump(fd, "/conf_number", conf_number, DumpMode::Overwrite);

  vector<vector<double>> rrr(NNN, vector<double>(2));
  int count = 0;
  for (int i = 0; i < Nblock; i++) {
    for (int j = 0; j < Nblock; j++) {
      for (int k = 0; k < nparticle(i, j); k++) {
        rrr[count][0] = rx(i, j, k) + Lblockx * i - rshiftx;
        rrr[count][1] = ry(i, j, k) + Lblocky * j - rshifty;
        while (rrr[count][0] < 0)
          rrr[count][0] += Lbox;
        while (rrr[count][1] < 0)
          rrr[count][1] += Lboy;
        while (rrr[count][0] > Lbox)
          rrr[count][0] -= Lbox;
        while (rrr[count][1] > Lboy)
          rrr[count][1] -= Lboy;

        count++;
      }
    }
  }

  config.write(rrr);
}

void write_ep(vector<float> &le, vector<float> &lp, vector<float> &lhv,
              string &write_file, int size_list) {

  File fd(write_file, File::ReadWrite);

  DataSet pset = fd.getDataSet("/pressure");

  unsigned int size = getSize(fd, "/pressure");

  pset.resize({size + size_list});
  for (unsigned int j = 0; j < size_list; j++) {
    pset.select({size + j}, {1}).write(lp[j]);
  }

  if (rpotential > 0) { // The following will be done when the interaction
                        // potential is WCA or IPL
    DataSet eset = fd.getDataSet("/energy");
    DataSet hvset = fd.getDataSet("/hypervirial");
    eset.resize({size + size_list});
    hvset.resize({size + size_list});

    for (unsigned int j = 0; j < size_list; j++) {
      eset.select({size + j}, {1}).write(le[j]);
      hvset.select({size + j}, {1}).write(lhv[j]);
    }
  } else {
    DataSet pyset = fd.getDataSet("/pressurey");
    pyset.resize({size + size_list});
    for (unsigned int j = 0; j < size_list; j++) {
      pyset.select({size + j}, {1}).write(lhv[j]);
    }
  }
}

void write_time(string &write_file, double cpu, double clock) {

  File fd(write_file, File::ReadWrite);

  auto t = time(nullptr); // now write the time as a string
  auto tm = *localtime(&t);

  ostringstream oss;
  oss << put_time(&tm, "%d-%m-%Y %H:%M:%S");
  string ds_name = "parameters/run_time_start";
  if (fd.exist(ds_name)) {
    ds_name = "parameters/run_time_finish";
    dump(fd, ds_name, oss.str());
  } else {
    dump(fd, ds_name, oss.str());
  }
  if (cpu != 0) {
    dump(fd, "parameters/cpu-time", cpu);
    dump(fd, "parameters/clock-time", clock);
  }
}
