import csv
import argparse
import matplotlib.pyplot as plt
import pyproj
import math
import decimal
import numpy as np 
from pyproj import Proj, transform

def read_csv(csv_filepath):
    csv_list = []
    with open(csv_filepath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            csv_list.append(row)
    return csv_list

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Visualisation')
    parser.add_argument('--input_gps', type=str, required=True, help='gps file')
    parser.add_argument('--input_state', type=str, required=True, help='state file')

    args = parser.parse_args()
    gps_file, state_file = args.input_gps, args.input_state
    gps_list, state_list = read_csv(gps_file), read_csv(state_file)
    print("Length: ", len(gps_list), len(state_list))
    lat_gps = [float(x[1]) for x in gps_list]
    lon_gps = [float(x[2]) for x in gps_list]
    alt_gps = [float(x[3]) for x in gps_list]
    lat_st = [float(x[1]) for x in state_list]
    lon_st = [float(x[2]) for x in state_list]
    alt_st = [float(x[3]) for x in state_list]
    proj = pyproj.Transformer.from_crs(4326, 3857, always_xy=True)

    x_coord_gps,y_coord_gps = proj.transform(lon_gps, lat_gps)
    x_coord_st,y_coord_st = proj.transform(lon_st, lat_st)

    min_x_coordinate = min(np.min(x_coord_gps), np.min(x_coord_st))
    min_y_coordinate = min(np.min(y_coord_gps), np.min(y_coord_st))
    x_coord_gps, x_coord_st = x_coord_gps-min_x_coordinate, x_coord_st-min_x_coordinate
    y_coord_gps, y_coord_st = y_coord_gps-min_y_coordinate, y_coord_st-min_y_coordinate

    plt.plot(x_coord_gps,y_coord_gps, 'r')
    plt.plot(x_coord_st,y_coord_st, 'b')
    plt.axis('equal')
    plt.show()
