import csv
import argparse
import matplotlib.pyplot as plt
import pyproj
import math
import decimal
import numpy as np 
from pyproj import Proj, transform
from matplotlib import animation 

from scipy.spatial.transform import Rotation as R

def read_csv(csv_filepath):
    csv_list = []
    with open(csv_filepath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            csv_list.append(row)
    return csv_list

def animation_frame(i):
    # print(x_coord_st[i], y_coord_st[i])
    line_gps.set_xdata(x_coord_gps[:i+1])
    line_gps.set_ydata(y_coord_gps[:i+1])

    ax.set_xlim(x_coord_gps[i]-10, x_coord_gps[i]+10)
    ax.set_ylim(y_coord_gps[i]-10, y_coord_gps[i]+10)
    moving_cum = 0

    while times_gps[i] > times_st[moving_cum]:
        moving_cum+=1

    line_st.set_xdata(x_coord_st[:moving_cum])
    line_st.set_ydata(y_coord_st[:moving_cum])

    return line_gps, line_st, 

def get_corners(pos, rot_mat, ego_size):
    w, l = ego_size
    x_corners = [w / 2, w / 2, -w/2, -w / 2]
    y_corners = [l / 2, -l / 2, -l / 2, l / 2]
    corners = np.array([x_corners, y_corners], dtype=np.float32)
    corners = rot_mat[:2, :2] @ corners
    corners += pos.reshape(2, 1)

    return corners.T

def draw_ego_car(sample_idx):
    sample_pos_x = x_coord_st[sample_idx]
    sample_pos_y = y_coord_st[sample_idx]
    pos = np.array([sample_pos_x, sample_pos_y])

    # Time, lla x3, gpi x 3, gvi x 3, quart x 4
    sample_quart = state_arr[sample_idx, 10:14]
    rot_mat = R.from_quat(sample_quart).as_matrix()
    corners_2d = get_corners(pos, rot_mat, ego_size=(60, 200))

    lines = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for p1_idx, p2_idx in lines:
        x_vals = [corners_2d[p1_idx, 0], corners_2d[p2_idx, 0]]
        y_vals = [corners_2d[p1_idx, 1], corners_2d[p2_idx, 1]]
        if p1_idx == 3 and p2_idx == 0:  # Heading
            plt.plot(x_vals, y_vals, 'c')
        else:
            plt.plot(x_vals, y_vals, 'g')

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Visualisation')
    parser.add_argument('--input_gps', type=str, required=True, help='gps file')
    parser.add_argument('--input_state', type=str, required=True, help='state file')

    args = parser.parse_args()
    gps_file, state_file = args.input_gps, args.input_state
    gps_list, state_list = read_csv(gps_file), read_csv(state_file)
    print("Length: ", len(gps_list), len(state_list))
    times_gps = [float(x[0]) for x in gps_list]
    lat_gps = [float(x[1]) for x in gps_list]
    lon_gps = [float(x[2]) for x in gps_list]
    alt_gps = [float(x[3]) for x in gps_list]
    times_st = [float(x[0]) for x in state_list]
    lat_st = [float(x[1]) for x in state_list]
    lon_st = [float(x[2]) for x in state_list]
    alt_st = [float(x[3]) for x in state_list]
    proj = pyproj.Transformer.from_crs(4326, 3857, always_xy=True)

    x_coord_gps,y_coord_gps = proj.transform(lon_gps, lat_gps)
    x_coord_st,y_coord_st = proj.transform(lon_st, lat_st)

    min_x_coord = min(x_coord_gps[0], x_coord_st[0])
    min_y_coord = min(y_coord_gps[0], y_coord_st[0])
    x_coord_gps, x_coord_st = [x-min_x_coord for x in x_coord_gps], [x-min_x_coord for x in x_coord_st]
    y_coord_gps, y_coord_st = [y-min_y_coord for y in y_coord_gps], [y-min_y_coord for y in y_coord_st]

    plt.style.use('dark_background')
    # print(x_coord_gps[0]), print(y_coord_gps[0])
    # print(np.max(x_coord_st))
    fig, ax = plt.subplots()
    ax.set_axisbelow(True)
    ax.grid()
    line_gps, = ax.plot(x_coord_gps[0], y_coord_gps[0], 'r', label="GPS")
    line_st, = ax.plot(x_coord_st[0], y_coord_st[0], 'b', label="State")
    # moving_timestamp = time_st[0]
    # print(times_gps)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)

    line_ani = animation.FuncAnimation(fig, func=animation_frame, frames = np.arange(0,10000,1), interval=10, blit=True)
    # plt.show()

    plt.legend()
    # line_ani.save('output/lines.mp4', writer=writer)
    # plt.grid()
    plt.show()

    # for idx in range(len(state_list)):
    #     timing = state_list[idx][0]
    #     plt.plot(x_coord_gps,y_coord_gps, 'r')
    #     plt.plot(x_coord_st,y_coord_st, 'b')
    # plt.axis('equal')
    # plt.savefig('output/full.png')
    # plt.show()
