#
#   Copyright (c) 2023, Yoshihiko Nishikawa, Werner Krauth, and A. C. Maggs
#
#   CUDA code for massively parallelized Monte Carlo simulation of
#   two-dimensional disks
#
#   URL: https://github.com/jellyfysh/SoftDisks
#   See LICENSE for copyright information
#
#   If you use this code or find it useful, please cite the following paper:
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
all: pressure.o  soft_particles_0th-smoothed.o

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


$(program_CU_OBJS): %.o: %.cu
	$(NVCC) -D$(POTCUDA) $(CUDAFLAGS) -c $< -o $@

$(program_CC_OBJS): %.o: %.cc
	$(CXX) $(CXXFLAGS) $(HDF5FLAGS) $(LIBS) -c $< -o $@

clean:
	$(RM)  *.o *dat 

clobber:clean
	$(RM) Soft_metropolis

html:
	kramdown <README.md  >readme.html

testpow:testpow.o
	$(NVCC) -o $@   testpow.o $(INCDIRS) $(LIBS) $(CXXFLAGS)
Metropolis_2d.o: ../CUDA_central/Soft.h 
pressure.o: ../CUDA_central/Soft.h 
