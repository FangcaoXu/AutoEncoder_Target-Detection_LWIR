import os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from tqdm import trange

days = 365
hours = range(6, 22, 2)
# hours = [2, 6, 10, 14, 18, 22]
# reflec = [5, 10, 30, 50, 80, 100]
reflec = []

keys = ['trans', 'grnd', 'down', 'up1_emission', 'up2_scatter', 'solar_single', 'solar_multi', 'total']

class BadCSV(Exception):
    """Raised when the shape size is unexpected or the csv file is malformed"""

def parse_csv(fileobj, header=1):
    try:
        # skip the first row and first column in the original table and then extract columns in angles
        # arr = np.genfromtxt(fileobj, skip_header=header, delimiter=",", autostrip=True,
        #                     usecols=range(skipcol, ncol + skipcol), dtype=np.float64)[:, 1:]
        # skip the first row and first column
        arr = np.genfromtxt(fileobj, skip_header= header, delimiter=",", autostrip=True, dtype=np.float64)[:, 1:]
    except:
        raise BadCSV("{} is broken".format(fileobj.name))
    finally:
        fileobj.close()
    # print("{} has the shape {}".format(fileobj.name, arr.shape))
    return arr

# tqdm(iterable) : Instantly make your loops show a smart progress meter
# trange is a short cut of tqdm(range())
def worker(data_dir, i, rows, cols):
    key = keys[i]
    # whole year
    arr=[]
    if len(reflec) == 0:
        file_patterns = {"trans": "TransmissionCSV/Radiance_{day}_{hh}_trans.csv",
                         "grnd": "GroundReflectedCSV/Radiance_{day}_{hh}_grnd.csv",
                         "down": "DownwellingCSV/Radiance_{day}_{hh}_down.csv",
                         "up1_emission": "PathThermalEmissionCSV/Radiance_{day}_{hh}_up1.csv",
                         "up2_scatter": "PathThermalScatteringCSV/Radiance_{day}_{hh}_up2.csv",
                         "solar_single": "SolarSingleScatteringCSV/Radiance_{day}_{hh}_solar1.csv",
                         "solar_multi": "SolarMultiScatteringCSV/Radiance_{day}_{hh}_solar2.csv",
                         "total": "TotalRadianceCSV/Radiance_{day}_{hh}_total.csv"}
        for d in trange(days, desc=key, position=i):  # d is from 0 to 364
            for h in hours:
                f = os.path.join(data_dir, file_patterns[key].format(day=d + 1, hh=h))
                arr.append(parse_csv(open(f)))
        return key, np.array(arr).reshape((days, len(hours), rows, cols))
    else:
        file_patterns = {"trans": "TransmissionCSV/Radiance_{day}_{hh}_{reflect}_trans.csv",
                         "grnd": "GroundReflectedCSV/Radiance_{day}_{hh}_{reflect}_grnd.csv",
                         "down": "DownwellingCSV/Radiance_{day}_{hh}_{reflect}_down.csv",
                         "up1_emission": "PathThermalEmissionCSV/Radiance_{day}_{hh}_{reflect}_up1.csv",
                         "up2_scatter": "PathThermalScatteringCSV/Radiance_{day}_{hh}_{reflect}_up2.csv",
                         "solar_single": "SolarSingleScatteringCSV/Radiance_{day}_{hh}_{reflect}_solar1.csv",
                         "solar_multi": "SolarMultiScatteringCSV/Radiance_{day}_{hh}_{reflect}_solar2.csv",
                         "total": "TotalRadianceCSV/Radiance_{day}_{hh}_{reflect}_total.csv"}
        for d in trange(days, desc=key, position=i):  # d is from 0 to 364
            for h in hours:
                for r in reflec:
                    # put the associated day, hour and reflectance to the file path
                    f = os.path.join(data_dir, file_patterns[key].format(day=d + 1, hh=h, reflect=r))
                    arr.append(parse_csv(open(f)))
        return key, np.array(arr).reshape((days, len(hours), len(reflec), rows, cols))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/home/graduate/fbx5002/disk10TB/DARPA/MatrixCSV/MidLatitude_VNIR_SWIR/DifferentMaterials/MODTRAN_OneYear")
    parser.add_argument("--subdatadir", default=False, action= "store_true")
    parser.add_argument("--out_dir", default="/home/graduate/fbx5002/disk10TB/DARPA/MachineLearningModels/trainModel_VNIR_SWIR")
    parser.add_argument("--rows", type=int, default=150)
    parser.add_argument("--cols", type=int, default=13)

    # parser.parse_args([]) # print the default values
    data_dir = parser.parse_args().data_dir
    out_dir = parser.parse_args().out_dir
    rows = parser.parse_args().rows
    cols = parser.parse_args().cols

    if parser.parse_args().subdatadir:
        dirs = os.listdir(data_dir)
        for dir in dirs:
            subdata_dir = os.path.join(data_dir, dir)
            data = dict()  # whole year
            with ProcessPoolExecutor(max_workers=8) as executor:
                # map(): apply the given function to each item of a give iterable
                # here the worker is the defined function above
                # ret[0] is the key, ret[1] is the first returned np.array
                for ret in executor.map(worker, [subdata_dir] * len(keys), range(len(keys)), [rows] * len(keys),
                                        [cols] * len(keys)):
                    if ret is not None:
                        data[ret[0]] = ret[1]
            out_file = os.path.join(out_dir, "data_{}_{}x{}.npz".format(dir, rows, cols))
            print("Saving data file {} ...".format(out_file))
            np.savez_compressed(out_file, **data)
            print("file saved!")
    else:
        data = dict()  # whole year
        with ProcessPoolExecutor(max_workers=8) as executor:
            for ret in executor.map(worker, [data_dir] * len(keys), range(len(keys)), [rows] * len(keys),
                                    [cols] * len(keys)):
                if ret is not None:
                    data[ret[0]] = ret[1]
        out_file = os.path.join(out_dir, "data_{}_{}x{}.npz".format(os.path.basename(data_dir), rows, cols))
        print("Saving data file {} ...".format(out_file))
        np.savez_compressed(out_file, **data)  # data.npz is numberpy zipped file
        print("file saved!")




