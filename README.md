
# **SoftDisks**

![license](https://img.shields.io/badge/license-GPLv3-brightgreen)

The SoftDisks package contains an implementation of a massively parallelized Monte Carlo algorithm [[Anderson, J. A., _et al._ (2013)](https://www.sciencedirect.com/science/article/abs/pii/S0021999113004968)]
for two-dimensional hard and soft disks written in C++ with CUDA extension for NVIDIA GPUs, and Python3 scripts for simple statistical analysis of simulation data. 
This package was used to produce the data presented in [[Nishikawa, Y., Krauth, W., and Maggs, A. C., (2023)](https://doi.org/10.1103/PhysRevE.108.024103)]. See also [our dataset on Zenodo](https://doi.org/10.5281/zenodo.7844567) for equilibrium configurations and the equations of state generated with the SoftDisks package.


## The model potentials

With the SoftDisks package, you can run a Monte Carlo simulation of the power-law interaction potential,

$$U(r) / \epsilon =
\begin{cases}
(\sigma/r)^m - (\sigma / r_c)^m 
& : r < r_c \\
0 & : r > r_c
\end{cases}$$

the Weeks-Chandler-Andersen (WCA) model,

$$U(r) / \epsilon =
\begin{cases}
4 \large[(\sigma/r)^{12} - (\sigma / r)^6 \large] + 1 
& : r < 2^{1/6} \sigma\\
0 & : r > 2^{1/6} \sigma,
\end{cases}$$

and the hard-disk model

$$U(r) / \epsilon =
\begin{cases}
\infty 
& : r < \sigma\\
0 & : r > \sigma.
\end{cases}$$

The interaction potential is controlled in a bash script file [`Scripts/Nishikawa2023/LoopRun`](Scripts/Nishikawa2023/LoopRun) as we will describe below.


## Requirements

In order to use the SoftDisk package, you need a computer with a NVIDIA GPU. The CUDA code also requires [the HDF5 library](https://www.hdfgroup.org/solutions/hdf5/) with a simple interface provided by [the HighFive library](https://bluebrain.github.io/HighFive/), and [the Boost C++ library](https://www.boost.org/). 
Note that our CUDA code currently does ***not*** work with NVIDIA CUDA >= 12.0, as some of the functions we have used in the code are depreciated in CUDA 12.0. We will update our code so that it works with the latest CUDA version in the future update.

The versions of the libraries and drivers for the CUDA code are the follwing:
- C++ compiler, e.g., g++
- NVIDIA CUDA Toolkit >= 10.0, < 12.0
- NVIDIA CUDA Driver >= 410.48
- HDF5 >= 1.10.4
- HighFive >= 2.2.2
- Boost >= 1.71.0
- ffmpeg >= 4.2.7

In addition to the driver and CUDA for NVIDIA GPU, you need the following Python3 packages:
- numpy >= 1.13.3
- scipy >= 1.6.0
- h5py >= 2.10.0
- matplotlib >= 3.5.1
- stresampling >= 1.0.2

## How to use

### Download
To get the package, manually clone the git repository from terminal.
```shell
git clone https://github.com/jellyfysh/SoftDisks.git
```

### Setting parameters for simulation
All the CUDA programs are controlled by a bash script [`Scripts/Nishikawa2023/LoopRun`](Scripts/Nishikawa2023/LoopRun), in which the following parameters need to be set.

- `Potential`: {'IPL', 'WCA', 'HS'} 

    This controls the interaction potential. It is either 'IPL' for the power-law interaction, 'WCA' for the WCA interaction, or 'HS' for the hard-sphere interaction. 

- `INIT`: {'RANDOM', 'RELOAD', 'REPLICATE'}

    The initial configuration is specified with this parameter. 
    
    - With 'RANDOM', the program first does a short simulation, starting from a crystalline configuration, at inverse temperature 10 times smaller than the one set by `Beta` explained below. The short run takes 1600 Monte Carlo sweeps per particle. 
    
        Note that, for hard disks where temperature is irrelevant, the resultant configuration may not be decorrelated well from the initial crystalline configuration.

    - With 'RELOAD', the simulation starts with the last configuration of the last run. This option thus fails if you launch the first run for a given parameter set.

    - With 'REPLICATE', 

- `SHAPE`: {'SQUARE', 'NONSQUARE'}

    The box shape of the simulation is controlled by this parameter. If you choose 'SQUARE', the simulation box is square, while with 'NONSQUARE', it is a rectangular box with ratio $1 : \sqrt{3} / 2$.

- `Density`: Real number

    The number density $N / V$, rather than the packing fraction.

- `Beta`: Real number

    Inverse temperature of the system. For the hard-sphere potential, this parameter does not change simulation results.

- `rpotential`: Interger number

    The exponent $m$ in the power-law model. This parameter is relevant only when `Potential` is 'IPL'. Note that `rpotential` cannot be an integer multiple of 612 (this number is reserveed for the WCA model in the CUDA code).

- `cutoff`: Real number

    The cutoff length $r_c / \sigma > 0$ in the power-law model, relevant only for the power-law model as well as `rpotential`.

- `Nrow`: Integer value

    The square root of the number of particles. Note that `Nrow` must be a power of 2, larger than or equal to 64.

- `MaxMCS`: Integer value

    This parameter controls the total number of Monte Carlo sweeps per particle. In the CUDA code, one loop corresponds to 160 sweeps, thus the total number is 160 times `MaxMCS`. Typical number is 10000000.


Besides those parameters, you also need to set `HIGHFIVE` manually in [`Scripts/SetPath`](Scripts/SetPath) file, which specifies the path to [the HighFive library](https://bluebrain.github.io/HighFive/) we mentioned above.

### How to run a simulation
After setting all the parameters above, run the following in [`Scripts/Nishikawa2023`](Scripts/Nishikawa2023) directory

```shell
bash ./LoopRun
```

## Simulation Results

For a given set of the parameters, a directory is created in [`Runs/`](Runs/) with a name depending on the parameters. 

- When `Potential` is 'IPL', it is

    `dens_${Density}_beta_${Beta}_Nrow_${Nrow}_rpot_${rpotential}_cutoff_${cutoff}${SUFFIX}`.

- When `Potential` is 'WCA' or 'HS', it is 

    `dens_${Density}_beta_${Beta}_Nrow_${Nrow}_Potential_${Potential}${SUFFIX}`.

For both cases, `SUFFIX` is none for `SHAPE` = 'SQUARE' and '_NONSQUARE' for `SHAPE` = 'NONSQUARE'.

All the simulation results are output into `data-*.h5` in [the HDF5 format](https://www.hdfgroup.org/solutions/hdf5/), in which you find several data sets and groups. You can read the contents of a hdf5 file from  the
command line  with `h5ls data-000.h5` for example.

After every simulation, a Python3 code [Scripts/Voronoi/Analysis.py](Scripts/Voronoi/Analysis.py) will automatically run to analyze the simulation results. You will find the following files after completion of the analysis.

- `*.sbm`: `*` = {`pressure`, `energy`}

    Each includes the mean and its 68% confidence interval of the pressure or the energy, estimated by [the stationary bootstrap method](https://github.com/YoshihikoNishikawa/stresampling).
- `*.sigma`: `*` = {`pressure`, `energy`}

    This file includes the number of blocking, the standard error, and its error, calculated by [the blocking method](http://aip.scitation.org/doi/10.1063/1.457480).

- `movie.webm`

    A movie produced by ffmpeg, showing the time evolution of the orientational field in real space.


## Citation

If you use the SoftDisks package or find it useful, please cite the following paper.

```bibtex
@article{PhysRevE.108.024103,
  title = {Liquid-hexatic transition for soft disks},
  author = {Nishikawa, Yoshihiko and Krauth, Werner and Maggs, A. C.},
  journal = {Phys. Rev. E},
  volume = {108},
  issue = {2},
  pages = {024103},
  numpages = {7},
  year = {2023},
  month = {Aug},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevE.108.024103},
  url = {https://link.aps.org/doi/10.1103/PhysRevE.108.024103}
}
```

## Contributing
If you wish to contribute, please submit a pull request.

If you find an issue or a bug, please contact [Yoshihiko Nishikawa](https://yoshihikonishikawa.github.io/) or raise an issue. 
