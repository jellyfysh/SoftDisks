#!/bin/bash
#
#   Copyright (c) 2023, Yoshihiko Nishikawa, Werner Krauth, and A. C. Maggs
#
#   Bash script for running Analysis.py
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
set -e
umask 0002
. ../SetPath

cd $DIR

pwd
export MPLBACKEND=agg

if [ -d ./cheap_picture ]; then
    rm -rf cheap_picture
fi

if [ -e ./movie.webm ]; then
    rm -f movie.webm
fi

mkdir cheap_picture

if [ ${SHAPE} = "NONSQUARE" ]; then
    export OPTION=--nonsquare
fi

if [ ${Potential} != "HS" ]; then
    ${PYTHON} ${TOP}/Scripts/Voronoi/Analysis.py ${OPTION}
elif [ ${Potential} == "HS" ]; then
    ${PYTHON} ${TOP}/Scripts/Voronoi/Analysis.py --hardsphere ${OPTION}
fi

printf "\e[31m video production \n \e[39m" $DIR
ffmpeg -i cheap_picture/cheap%05d.png -threads 16 -crf 52 -c:v libvpx-vp9 -b:v 0 movie.webm
printf "\e[31m video done\n \e[39m" $DIR
rm -rf cheap_picture
