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
#include <sys/resource.h>
#include <sys/time.h>

class Timer {
private:
  struct timeval t1, t2;
  struct timezone tz;
  rusage tr1, tr2;

public:
  Timer() {
    gettimeofday(&t1, &tz);
    getrusage(RUSAGE_SELF, &tr1);
    cout << "CPU\t" << (tr1.ru_utime.tv_sec + tr1.ru_utime.tv_usec / 1000000.)
         << "\t" << tr1.ru_stime.tv_sec + tr1.ru_utime.tv_usec / 1000000.
         << endl;
  }

  double clock() {
    gettimeofday(&t2, &tz);
    return (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000000.;
  }
  double cpu() {
    getrusage(RUSAGE_SELF, &tr2);
    return (tr2.ru_utime.tv_sec + tr2.ru_utime.tv_usec / 1000000.);
  }

  void stats() {
    gettimeofday(&t2, &tz);
    cout << "Clock Time \t"
         << (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000000.
         << endl;
    getrusage(RUSAGE_SELF, &tr2);
    cout << "CPU\t" << (tr2.ru_utime.tv_sec + tr2.ru_utime.tv_usec / 1000000.)
         << "\t" << tr2.ru_stime.tv_sec + tr2.ru_utime.tv_usec / 1000000.
         << endl;
    cout << "Diff "
         << (tr2.ru_utime.tv_sec + tr2.ru_utime.tv_usec / 1000000.) -
                (tr1.ru_utime.tv_sec + tr1.ru_utime.tv_usec / 1000000.)
         << endl;
  }
  void tic() {
    ofstream oo;
    oo.open("tic.toc");
    getrusage(RUSAGE_SELF, &tr2);
    oo << (tr2.ru_utime.tv_sec + tr2.ru_utime.tv_usec / 1000000.) -
              (tr1.ru_utime.tv_sec + tr1.ru_utime.tv_usec / 1000000.)
       << endl;
    oo.close();
  }
};
