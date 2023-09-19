#
#   Copyright (c) 2023, Yoshihiko Nishikawa, Werner Krauth, and A. C. Maggs
#
#   Python3 code for replicating a particle configuration to double its system size
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
import h5py  # hdf5
import numpy as np
import glob
from functools import partial
import argparse

### argparser ###
parser = argparse.ArgumentParser(
    description="Python code for calculation of pressure, voronoi, and orientational order"
)
parser.add_argument("-ns", "--nonsquare", action="store_true")
args = parser.parse_args()


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


def main():
    files = sorted(glob.glob("data-*.h5"))
    for filename in files:
        f = h5py.File(filename, "r")  # Find the hdf5 file for the lastest run

    conf_number = extract_item(f, "conf_number")
    print("conf_number", conf_number)
    Nblock = extract_item(f, "Nblock")
    Nrow = extract_item(f, "Nrow")
    density = extract_item(f, "density")
    if not args.nonsquare:
        Lblockx = Nrow / np.sqrt(density) / Nblock
        Lblocky = Lblockx
    else:
        Lblockx = (Nrow / density**0.5) * (2.0 / 3.0**0.5) ** 0.5 / Nblock
        Lblocky = 3.0**0.5 * 0.5 * Lblockx

    FLAG = True
    count = 0
    while FLAG:
        print(count)
        config = f["config-" + str(conf_number - 1 - count)]
        config = config[:]  # extract  array from hdf5 data structure
        config = np.array(config)
        one_block = np.array([Lblockx, Lblocky])
        index = (np.floor(config / one_block)).astype(np.int64)
        print(index)
        print(index.dtype)

        if Nblock in index:
            count = count + 1
        else:
            config = config - one_block * index
            FLAG = False

    plist = {
        "/parameters/",  # copy over these variables
        "/conf_number",
    }

    g = h5py.File("start.h5", "w")  # write to file

    print("config found:\t", conf_number - 1 - count, config.shape, config.dtype)

    dummy = config
    # +x shift
    shiftx = index + np.array([[Nblock, 0] for i in range(Nrow * Nrow)], dtype=np.int64)
    # +y shift
    shifty = index + np.array([[0, Nblock] for i in range(Nrow * Nrow)], dtype=np.int64)
    # +x+y shift
    shiftxy = index + np.array(
        [[Nblock, Nblock] for i in range(Nrow * Nrow)], dtype=np.int64
    )

    config = np.append(config, dummy, axis=0)
    config = np.append(config, dummy, axis=0)
    config = np.append(config, dummy, axis=0)

    index = np.append(index, shiftx, axis=0)
    index = np.append(index, shifty, axis=0)
    index = np.append(index, shiftxy, axis=0)

    ds_name = "/config"
    g.create_dataset(ds_name, data=config)
    ds_name = "/index"
    g.create_dataset(ds_name, data=index)

    for item in plist:
        x = f[item]
        print(item, ":\t", x)
        f.copy(item, g)

    del g["/parameters/hostname"]
    del g["/parameters/username"]


if __name__ == "__main__":
    main()
