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
set -e     # exit on error
umask 0002 # files can be modified by GPU user group

if [ ${SHAPE} = "NONSQUARE" ]; then
    export SUFFIX=_NONSQUARE
fi

printf "\e[31m Analysis in: %s \n \e[39m" $DIR

${SCRIPTS}/../Voronoi/Analysis.sh

printf "\e[31m Analysis done: %s \n \e[39m" $DIR
