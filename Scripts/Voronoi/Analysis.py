#
#   Copyright (c) 2023, Yoshihiko Nishikawa, Werner Krauth, and A. C. Maggs
#
#   Python3 code for analyzing the orientational order of two-dimensional disks, using the Voronoi tessellation
#
#   URL: https://github.com/
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
import sys
import h5py  # hdf5
import math
import scipy
import scipy.fft
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.font_manager
from scipy.spatial import Delaunay
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.cm as cm
import argparse
from stresampling import stationary_bootstrap as sbm

from functools import partial
import scipy.ndimage

#### Setting fonts ####
del matplotlib.font_manager.weight_dict["roman"]
matplotlib.font_manager._load_fontmanager(try_read_cache=False)
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"

### argparser ###
parser = argparse.ArgumentParser(
    description="Python code for calculation of pressure, voronoi, and orientational order"
)
parser.add_argument("-hs", "--hardsphere", action="store_true")
parser.add_argument("-ns", "--nonsquare", action="store_true")
args = parser.parse_args()


SAVE = 1  # save the figures
NORMALIZE = 0
HEXFIG = 1  # produce a picture of the local orientational order parameter from the final configuration
GRID_SCALE = 8  # parameter for density-density calculation

#  variables for the wilson code
wilson_nblock = 0
wilson_result = []
wilson_var = 0


def find_item(name, target):
    if target in name:
        return name


def extract_item(file, target):
    """Extract the value of an item in a hdf5 file
    Args:
        file: The path to the hdf5 file
        target: The name of the dataset to be found
    Returns:
        output: The value of the item if found
    """
    path = file.visit(partial(find_item, target=target))
    output = file[path]
    output = output[()]
    return output


def read_timeseries(variable):
    """Reading data files
    Args:
        variable: The name of the variable to be read from data files
    Returns:
        p: A timeseries of the variable
    """
    files = sorted(glob.glob("data-*.h5"))
    p = np.empty(0)
    for filename in files:
        f = h5py.File(filename, "r")
        g = extract_item(f, variable)
        g = g[:]
        p = np.append(p, g)

    p = p[p.shape[0] // 6 :]  # discard first part of array
    np.savetxt(str(variable) + ".txt", p)
    print(p.shape)
    return p


def decimate(v, variable):
    c0(v, variable)
    # average over blocks of length 2
    v = v[: (2 * (v.shape[0] // 2))].reshape((-1, 2))
    v2 = np.mean(v, axis=1)  # axis tells us which direction we average over
    if v2.shape[0] > 1:
        decimate(v2, variable)


def c0(v, variable):
    global wilson_nblock
    global wilson_var
    sm = np.mean(v)
    s2 = np.std(v)

    if wilson_nblock == 0:
        wilson_var = s2
        # print(line, file=sys.stderr)
    s2 = s2 / math.sqrt((v.shape[0] - 1))
    sigma = s2 / math.sqrt(2 * (v.shape[0] - 1))
    wilson_result.append((wilson_nblock, s2, sigma))
    wilson_nblock = wilson_nblock + 1


def calculate_wilson(variable):
    mode = 1  # mode=1 calculates errors, mode=3 autocorrelation times
    global wilson_nblock
    global wilson_result
    global wilson_var
    wilson_nblock = 0
    wilson_result = []
    wilson_var = 0

    v = read_timeseries(variable)
    print(v)
    stat = calculate_stationary_bootstrap(v)

    decimate(v, variable)
    scaled = []
    lines = np.empty(0)
    for r in wilson_result:
        ix, z, e = r
        if mode == 2:
            z = z / wilson_var
            e = e / wilson_var
        if mode == 3:
            e = e * z * v.shape[0] / wilson_var / wilson_var
            z = z * z * v.shape[0] / wilson_var / wilson_var
        scaled.append((ix, z, e))

        line = "{0}\t{1:.8g}\t{2:.8g}".format(ix, z, e)

        lines = np.append(lines, line)

    plot_wilson(scaled, mode, stat, variable)

    output_line = np.array(
        ["Mean: " + str(stat.mean) + " error: " + str(stat.mean - stat.low)]
    )
    np.savetxt(str(variable) + ".sbm", output_line, fmt="%s")
    np.savetxt(str(variable) + ".sigma", lines, fmt="%s")


def calculate_stationary_bootstrap(timeseries):
    stat = sbm.conf_int(timeseries, np.mean, 0.68, method="symbt")
    return stat


def plot_wilson(r, mode, stat, variable):
    arr = np.array(r)
    plt.figure(400)
    plt.clf()
    plt.errorbar(
        arr[:, 0],
        arr[:, 1],
        yerr=arr[:, 2],
        fmt="ko",
        fillstyle="none",
        elinewidth=1,
        capsize=3,
    )  # Plot some data on the axes
    plt.xlabel("block", fontsize=15)
    plt.ylabel(r"estimated error", fontsize=15)
    plt.yscale("log")
    plt.hlines(
        stat.mean - stat.low,
        xmax=arr[-1, 0],
        xmin=arr[0, 0],
        linestyles="dashed",
        label="Stationary bootstrap",
    )
    plt.title("wilson blocking mode =" + str(mode) + ", " + variable)
    plt.savefig("wilson_" + variable + ".pdf", bbox_inches="tight")


def hexatic(periodic_config):
    # work with extended set of particles, to avoid edge corrections
    """Calculate the orientational order parameter from a configuration
    Args:
        periodic_config: An extended particle configuration
    Returns:
        local_psi6: A numpy array of the local psi6 for particles
        index_mapping: Color map index
        triangles: Delaunay triangulation
    """
    npart = periodic_config.shape[0]

    # local orientational order
    local_psi6 = np.zeros(npart, dtype="complex128")
    counter = np.zeros(npart)  # number of links to site
    triangles = Delaunay(periodic_config)

    p0 = triangles.simplices[:, 0]  # 2N triangles, 3 links per triangle
    p1 = triangles.simplices[:, 1]
    p2 = triangles.simplices[:, 2]

    # 2 N vectors * 3, each link is twice here
    d01 = periodic_config[p0, :] - periodic_config[p1, :]
    d20 = periodic_config[p2, :] - periodic_config[p0, :]
    d12 = periodic_config[p1, :] - periodic_config[p2, :]

    theta01 = 6 * np.arctan2(d01[:, 1], d01[:, 0])  # 2 N angles
    theta20 = 6 * np.arctan2(d20[:, 1], d20[:, 0])  #
    theta12 = 6 * np.arctan2(d12[:, 1], d12[:, 0])  # making 6N contributions

    # phase on the three sides of each triangle
    expt01 = np.exp(complex(0, 1) * theta01)
    expt20 = np.exp(complex(0, 1) * theta20)
    expt12 = np.exp(complex(0, 1) * theta12)

    np.add.at(local_psi6, p0, expt01)  # distribute phase from links to 2 nodes
    np.add.at(local_psi6, p1, expt01)

    np.add.at(local_psi6, p2, expt20)
    np.add.at(local_psi6, p0, expt20)

    np.add.at(local_psi6, p1, expt12)
    np.add.at(local_psi6, p2, expt12)

    np.add.at(counter, p0, 2)  # counter number of links to node
    np.add.at(counter, p1, 2)
    np.add.at(counter, p2, 2)

    # normalize local orientational order by the number of neighbors
    local_psi6 = np.divide(local_psi6, counter)
    index_mapping = (
        (np.angle(local_psi6) + math.pi) / 2.0 / math.pi
    )  # colormap index in [0 1]
    return local_psi6, index_mapping, triangles


def inputs(num, filename):
    """Read data from hdf5 filesystem, first call to write information used in latex

    Args:
        num: The index of the configuration to be read
        filename: The path to the hdf5 file

    Returns:
        config: The particle configuration
        rho: number density
    """
    f = h5py.File(filename, "r")

    rho = extract_item(f, "density")
    config = extract_item(f, "config-" + str(num))

    return config, rho


def cheap_voronoi_plot(vq, nrec):
    """Make a plot of local orientational orders
    Args:
        vq: A gridded orientational order
        nrec: The index of the configuration
    Returns:
        None
    """

    plt.figure(700)
    plt.clf()
    index = (np.angle(vq) + math.pi) / 2.0 / math.pi  # colormap index in [0 1]
    ax = plt.gca()
    colormap = ax.pcolor(index, cmap=cm.hsv)

    aspect = 1
    if args.nonsquare:
        aspect = 3.0**0.5 * 0.5
    if SAVE:
        s = str(nrec).zfill(5)
        ax.set_aspect(aspect)
        ax.axis("off")
        plt.savefig("cheap_picture/cheap" + s + ".png", bbox_inches="tight")


# take local orientational order to a regular grid and calculate power spectrum and pair correlation
def gridding(grid, triangles, local_psi6, Lbox):
    """Make a grid of the local orientational orders in space
    Args:
        grid: The grid to be used for gridding
        triangles: The Delaunay tessellation of the configuration
        local_psi6: An array of the local orientatinal order for particles
        Lbox: The linear length of the box
    Returns:
        vq: A gridded orientational order in space
    """
    x = np.arange(Lbox / 2.0 / grid, Lbox, Lbox / grid)
    if not args.nonsquare:
        y = x
    else:
        yLbox = 3.0**0.5 * 0.5 * Lbox
        y = np.arange(yLbox / 2.0 / grid, yLbox, yLbox / grid)

    xx, yy = np.meshgrid(x, y)
    interp = scipy.interpolate.LinearNDInterpolator(triangles, local_psi6)
    vq = interp(xx, yy)

    return vq


def plot_voronoi(periodic_config, Lbox, index):  # Visualize local orientational order
    """Make a plot of the local orientaional order with full Voronoi tessellation
    Args:
        periodic_config: A particle configuration
        Lbox: The linear length of the box
        index: Color map index
    Returns:
        None
    """
    vor = Voronoi(periodic_config)
    plt.figure(502)
    axx = plt.gca()
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1, clip=True)
    # or cmap=cmocean.cm.phase
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.hsv)
    voronoi_plot_2d(
        vor, show_points=False, show_vertices=False, line_width=0, s=1, ax=axx
    )
    for r in range(len(vor.point_region)):
        region = vor.regions[vor.point_region[r]]
        if not -1 in region:
            polygon = [vor.vertices[i] for i in region]
            plt.fill(*zip(*polygon), color=mapper.to_rgba(index[r]))

    if not args.nonsquare:
        yLbox = Lbox
    else:
        yLbox = 3.0**0.5 * 0.5 * Lbox
    ax = plt.gca()
    ax.set_aspect("equal")
    ax.set_xlim(0, Lbox)
    ax.set_ylim(0, yLbox)
    plt.title("Full Voronoi calculation")
    if SAVE:
        plt.savefig("hexaticfield.png", bbox_inches="tight", dpi=200)
    return None


def extend(config, Lbox, N):
    """Add periodic images for Voronoi, trimmed to distance "edge"
    Args:
        config: Particle configuration
        Lbox: Linear size of the box
        N: Number of particles
    Returns:
        per: Extended particle configuration
    """

    edge = 2 * Lbox / math.sqrt(N)
    if args.nonsquare:
        yLbox = 3.0**0.5 * 0.5 * Lbox
    else:
        yLbox = Lbox

    periodic_config = np.concatenate(
        (
            config,
            config + [Lbox, 0],
            config + [0, yLbox],
            config + [Lbox, yLbox],
            config + [-Lbox, 0],
            config + [0, -yLbox],
            config + [-Lbox, -yLbox],
            config + [Lbox, -yLbox],
            config + [-Lbox, yLbox],
        )
    )
    inx = (
        (periodic_config[:, 1] > -edge)
        & (periodic_config[:, 0] > -edge)
        & (periodic_config[:, 1] < yLbox + edge)
        & (periodic_config[:, 0] < Lbox + edge)
    )

    # particles with a duplicated shell of width = edge, used for local-orientational-order calculation
    periodic_config = periodic_config[inx, :]
    return periodic_config


def do_loop():  # loop over saved configurations stored on disk
    """Calculate the orientational order for configurations and output pictures
    Args:
        None
    Return:
        None
    """

    filename = "data-000.h5"
    inputs(0, filename)  # open file

    files = sorted(glob.glob("data-*.h5"))
    count = 0
    total_samples = 0
    for filename in files:
        f = h5py.File(filename, "r")
        conf_number = extract_item(f, "conf_number")
        total_samples = total_samples + conf_number

    for filename in files:
        f = h5py.File(filename, "r")
        conf_number = extract_item(f, "conf_number")
        finish = conf_number
        print(filename)

        for nrec in range(0, finish, 1):
            print("recording:\t", count, "/", total_samples)
            config, rho = inputs(nrec, filename)
            n = max(config.shape)
            if not args.nonsquare:
                Lbox = math.sqrt(n / rho)  # box size
            else:
                Lbox = math.sqrt(n / rho) * (2.0 / 3.0**0.5) ** 0.5  # box size

            # gridding for local orientational order
            grid = math.floor(math.sqrt(n))
            periodic_config = extend(config, Lbox, n)

            local_psi6, index_mapping, triangles = hexatic(
                periodic_config
            )  # local orientational order

            # local orientational order on grid and fft
            vq = gridding(grid, triangles, local_psi6, Lbox)

            # png files to cheap_picture/ for video production
            cheap_voronoi_plot(vq, count)

            # if (nrec == finish - 1) and HEXFIG:  # high quality image from the last configuration
            # high quality image from the last configuration
            if (count == total_samples - 1) and HEXFIG:
                # plot local orientational order, very slow
                plot_voronoi(periodic_config, Lbox, index_mapping)

            count = count + 1


def main():
    if args.nonsquare:
        print("Analysis for non-square box")

    if not args.hardsphere:
        calculate_wilson("energy")
        calculate_wilson("hypervirial")

    calculate_wilson("pressure")
    do_loop()


if __name__ == "__main__":
    main()
