import csv
import argparse
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.lines import Line2D

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
    corners_2d = get_corners(pos, rot_mat, ego_size=(2, 4))
    head_x_1, head_y_1 = [], []
    head_x_2, head_y_2 = [], []

    lines = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for p1_idx, p2_idx in lines:
        x_vals = [corners_2d[p1_idx, 0], corners_2d[p2_idx, 0]]
        y_vals = [corners_2d[p1_idx, 1], corners_2d[p2_idx, 1]]
        if p1_idx == 3 and p2_idx == 0:  # Heading
            ax.plot(x_vals, y_vals, 'c')
            head_x_1.append(x_vals)
            head_y_1.append(y_vals)
        else:
            ax.plot(x_vals, y_vals, 'g')
            head_x_2.append(x_vals)
            head_y_2.append(y_vals)
    return head_x_1, head_y_1, head_x_2, head_y_2


def animation_frame(i):
    # print(x_coord_st[i], y_coord_st[i])
    line_gps.set_xdata(x_coord_gps[:i+1])
    line_gps.set_ydata(y_coord_gps[:i+1])

    ax.set_xlim(x_coord_gps[i]-10, x_coord_gps[i]+10)
    ax.set_ylim(y_coord_gps[i]-10, y_coord_gps[i]+10)

    moving_cum = 0
    while times_gps[i] > times_st[moving_cum]:
        moving_cum += 1
    head_x_1, head_y_1, head_x_2, head_y_2 = draw_ego_car(moving_cum-1)
    ax.get_lines().pop(-1).remove()
    ax.get_lines().pop(-1).remove()
    ax.get_lines().pop(-1).remove()
    ax.get_lines().pop(-1).remove()
    if i > 0:
        ax.get_lines().pop(-1).remove()
        ax.get_lines().pop(-1).remove()
    line_head_1 = Line2D(head_x_1, head_y_1, color='c')
    line_head_2 = Line2D(head_x_2, head_y_2, color='g')
    ax.add_line(line_head_1)
    ax.add_line(line_head_2)

    line_st.set_xdata(x_coord_st[:moving_cum])
    line_st.set_ydata(y_coord_st[:moving_cum])

    return line_gps, line_st, line_head_1, line_head_2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualisation')
    parser.add_argument('--input_gps', type=str, required=True, help='gps file')
    parser.add_argument('--input_state', type=str, required=True, help='state file')

    args = parser.parse_args()
    gps_file, state_file = args.input_gps, args.input_state
    gps_arr, state_arr = read_csv(gps_file), read_csv(state_file)
    print("Shape - gps_arr: {}, state_arr: {}".format(gps_arr.shape, state_arr.shape))

    times_gps, x_coord_gps, y_coord_gps = gps_arr[:, 0], gps_arr[:, -3], gps_arr[:, -2]

    times_st, x_coord_st, y_coord_st = state_arr[:, 0], state_arr[:, -3], state_arr[:, -2]

    plt.style.use('dark_background')

    fig, ax = plt.subplots()
    ax.set_axisbelow(True)
    ax.grid()
    line_gps, = ax.plot(x_coord_gps[0], y_coord_gps[0], 'r', label="GPS")
    line_st, = ax.plot(x_coord_st[0], y_coord_st[0], 'b', label="State")

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=2500)

    line_ani = animation.FuncAnimation(fig, func=animation_frame, frames=np.arange(
        0, 10000, 1), interval=20, blit=False, repeat=False, save_count=200)

    plt.legend()
    # line_ani.save('output/lines_enu_20210620.mp4', writer=writer)

    # plt.grid()
    plt.show()
