#!/bin/bash
#
#   Copyright (c) 2023, Yoshihiko Nishikawa, Werner Krauth, and A. C. Maggs
#
#   Bash script for running the CUDA code
#
#   URL: https://github.com/jellyfysh/SoftDisks
#   See LICENSE for copyright information
#
#   If you use this code or find it useful, please cite the following paper
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
set -e                                                                                                                                                 # exit on error
umask 0002                                                                                                                                             # so all gpu members can use files
. ../SetPath                                                                                                                                           # set directory paths
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader | nl -v 0 | sort -nrk 2 | cut -f 1 | head -n 1 | xargs) # This automatically finds the most free gpus and assign the run to one of them.

export POTCUDA=SOFTCUDA
MAKEFLAG=Soft_metropolis
if [ ${Potential} = "HS" ]; then
    POTCUDA=HARDCUDA
    MAKEFLAG=HS_metropolis
fi

(
    cd ${TOP}/CUDA${SUFFIX}/
    make clean
    make version.h
    make $MAKEFLAG -j 10
)

if [ ${INIT} = "RANDOM" ]; then
    mkdir $DIR #will fail if already exists due to "set -e"
    cd $DIR
elif [ ${INIT} = "REPLICATE" ]; then
    echo $DIR
    mkdir $DIR #will fail if already exists due to "set -e"
    SDIR=${RUNDIR}/dens_${Density}_beta_${Beta}_Nrow_$((Nrow / 2))_rpot_${rpotential}_cutoff_${cutoff}${SUFFIX}
    if [ ${Potential} != "IPL" ]; then
        SDIR=${RUNDIR}/dens_${Density}_beta_${Beta}_Nrow_$((Nrow / 2))_Potential_${Potential}${SUFFIX}
    fi

    cd $SDIR #will fail if not existing
    ${PYTHON} ${SCRIPTS}/Extract/replicate.py ${OPTION}
    mv start.h5 $DIR
    cd $DIR
elif [ ${INIT} = "RELOAD" ]; then
    # Make start.h5
    cd $DIR
    ${PYTHON}${SCRIPTS}/Extract/extract.py ${OPTION}
fi

printf "\e[31m running in %s  \e[39m\n" $DIR

if [ ${Potential} = "HS" ]; then
    cp ${TOP}/CUDA${SUFFIX}/HS_metropolis $DIR
    (time ./HS_metropolis $Density $Beta 2>>error.log 1>>run.log) 2>>time.log
else
    cp ${TOP}/CUDA${SUFFIX}/Soft_metropolis $DIR
    (time ./Soft_metropolis $Density $Beta 2>>error.log 1>>run.log) 2>>time.log
fi
printf "\e[31m Finished in %s  \e[39m\n" $DIR

\rm -f parameters.txt
touch parameters.txt
for i in data*h5; do
    echo $i >>parameters.txt
    h5ls -d $i/parameters | grep -v Data: >>parameters.txt
done
