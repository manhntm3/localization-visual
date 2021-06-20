import csv
import argparse
import matplotlib.pyplot as plt
import numpy as np

from scipy.spatial.transform import Rotation as R


def read_csv(csv_filepath):
    csv_list = []
    with open(csv_filepath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            csv_list.append(row)
    return np.array(csv_list, dtype=np.float64)


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualisation')
    parser.add_argument('--input_gps', type=str, required=True, help='gps file')
    parser.add_argument('--input_state', type=str, required=True, help='state file')

    args = parser.parse_args()
    gps_file, state_file = args.input_gps, args.input_state
    gps_arr, state_arr = read_csv(gps_file), read_csv(state_file)
    print("Shape - gps_arr: {}, state_arr: {}".format(gps_arr.shape, state_arr.shape))
    lat_gps = gps_arr[:, 1]
    lon_gps = gps_arr[:, 2]
    alt_gps = gps_arr[:, 3]
    lat_st = state_arr[:, 1]
    lon_st = state_arr[:, 2]
    alt_st = state_arr[:, 3]

    x_coord_gps, y_coord_gps = gps_arr[:, -3], gps_arr[:, -2]
    x_coord_st, y_coord_st = state_arr[:, -3], state_arr[:, -2]

    # Start at (0, 0)
    min_x_coordinate = np.array(x_coord_gps[0])
    min_y_coordinate = np.array(y_coord_gps[0])

    x_coord_gps, x_coord_st = x_coord_gps-min_x_coordinate, x_coord_st-min_x_coordinate
    y_coord_gps, y_coord_st = y_coord_gps-min_y_coordinate, y_coord_st-min_y_coordinate

    plt.plot(x_coord_gps, y_coord_gps, 'r')
    plt.plot(x_coord_st, y_coord_st, 'b')

    gpi_st_x = state_arr[:, 4]
    gpi_st_y = state_arr[:, 5]

    print('diff x: {:.4f}, y: {:.4f}'.format((gpi_st_x - x_coord_st).max(), (gpi_st_y - y_coord_st).max()))

    plt.plot(gpi_st_x, gpi_st_y, 'g')


    # Take a sample position, then plot car direction
    draw_ego_car(sample_idx=20000)

    plt.axis('equal')
    plt.show()
