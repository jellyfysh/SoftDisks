#
#   Copyright (c) 2023, Yoshihiko Nishikawa, Werner Krauth, and A. C. Maggs
#
#   CUDA code for massively parallelized Monte Carlo simulation of
#   two-dimensional disks
#
#   URL: https://github.com/jellyfysh/SoftDisks
#   See LICENSE for copyright information
#
#  	If you use this code or find it useful, please cite the following paper:
#
#   @article{PhysRevE.108.024103,
#       title = {Liquid-hexatic transition for soft disks},
#       author = {Nishikawa, Yoshihiko and Krauth, Werner and Maggs, A. C.},
#       journal = {Phys. Rev. E},
#       volume = {108},
#       issue = {2},
#       pages = {024103},
#       numpages = {7},
#       year = {2023},
#       month = {Aug},
#       publisher = {American Physical Society},
#       doi = {10.1103/PhysRevE.108.024103},
#       url = {https://link.aps.org/doi/10.1103/PhysRevE.108.024103}
#   }
#
#
all:HS_metropolis  Soft_metropolis

CUDA_central:.FORCE version.h
	cd CUDA_central &&  $(MAKE)  all

CUDA_extra_HS:.FORCE version.h
	cd CUDA_extra_HS && $(MAKE)  all

CUDA_extra_soft:.FORCE version.h
	cd CUDA_extra_soft &&  $(MAKE)  all

.FORCE:

program_CC_SRCS := $(wildcard *.cc)
program_CC_OBJS := ${program_CC_SRCS:.cc=.o}

program_CU_SRCS := $(wildcard *.cu)
program_CU_OBJS := ${program_CU_SRCS:.cu=.o}


CXX=g++
NVCC=nvcc
CXXFLAGS=  -DNrow=$(Nrow) -DMaxMCS=$(MaxMCS) -Drpotential=$(rpotential) -Dcutoff=$(cutoff) -use_fast_math -O3  -std=c++14 
HDF5FLAGS = -I$(HIGHFIVE)/include -I/opt/local/include -I/usr/include/hdf5/serial

CUDAFLAGS= -g   --device-c -rdc=true  $(CXXFLAGS) # -gencode=arch=compute_70,code=compute_70  $(CXXFLAGS) 
LIBS=  -lpthread -lcudart -lcublas -L/usr/lib/x86_64-linux-gnu/hdf5/serial  -I/usr/include/hdf5/serial -lhdf5  -lboost_system -lm


HS_metropolis: CUDA_central CUDA_extra_HS
	$(NVCC) -o $@   CUDA_central/*.o  CUDA_extra_HS/*.o  $(LIBS) $(CXXFLAGS)

Soft_metropolis: CUDA_central CUDA_extra_soft
	$(NVCC) -o $@   CUDA_central/*.o  CUDA_extra_soft/*.o  $(LIBS) $(CXXFLAGS)


clean:
	$(RM)  *.o *dat version.h
	cd CUDA_central && $(MAKE)  clean
	cd CUDA_extra_HS  &&  $(MAKE) clean
	cd CUDA_extra_soft  && $(MAKE) clean


html:
	kramdown <README.md  >readme.html

version.h:
	touch version.h
	git log -n 1 --format=format:"#define GIT_COMMIT \"%H\"%n" HEAD > $@


global.cu:Soft.h

init.o: Soft.h
main.o: Soft.h output_data.h
pressure.o: Soft.h
shift.o: Soft.h
shiftGPU.o: Soft.h
simulate.o: Soft.h
dump_data.o: output_data.h
Metropolis_2d.o: Soft.h fastpow.h
