import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from Low_Pass_Filter import *


def plot_marker_xyz(trc_path: str,
                    marker_name: str,
                    filter_name: str = None):
    """
    .trc 파일과 마커 이름, 그리고 optional filter_name을 받아
    시간–X, 시간–Y, 시간–Z 궤적을 3개의 subplot으로 한 번에 출력
    
    filter_name: None | 'butterworth' | 'median' | 'bessel'
    """
    time = extract_time_numpy(trc_path)
    if filter_name is None:
        x, y, z = extract_marker_numpy(trc_path, marker_name)
        title_suffix = "(Original)"
    else:
        if filter_name == 'butterworth':
            x, y, z = smooth_marker_butterworth(trc_path, marker_name)[0:3]
        elif filter_name == 'median':
            x, y, z = smooth_marker_median(trc_path, marker_name)[0:3]
        elif filter_name == 'bessel':
            x, y, z = smooth_marker_bessel(trc_path, marker_name)[0:3]
        else:
            raise ValueError(f"Unknown filter: {filter_name}")
        title_suffix = f" (Filter: {filter_name})"

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(time, x, linewidth=1.5)
    axes[0].set_ylabel('X Position (m)')
    axes[0].set_title(f'{marker_name} X over Time{title_suffix}')
    axes[0].grid(True)
    axes[1].plot(time, y, linewidth=1.5)
    axes[1].set_ylabel('Y Position (m)')
    axes[1].set_title(f'{marker_name} Y over Time{title_suffix}')
    axes[1].grid(True)
    axes[2].plot(time, z, linewidth=1.5)
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Z Position (m)')
    axes[2].set_title(f'{marker_name} Z over Time{title_suffix}')
    axes[2].grid(True)

    fig.suptitle(f'{marker_name} Trajectory {title_suffix}', fontsize=16)
    plt.tight_layout()
    #plt.show()


def animate_marker_3d(trc_path: str,
                      marker_name: str,
                      interval: int = 50,
                      filter_name: str = None) -> FuncAnimation:
    """
    단일 마커의 3D 궤적 애니메이션
    filter_name: None | 'butterworth' | 'median' | 'bessel'
    """
    t = extract_time_numpy(trc_path)
    if filter_name is None:
        x, y, z = extract_marker_numpy(trc_path, marker_name)
    else:
        if filter_name == 'butterworth':
            x, y, z = smooth_marker_butterworth(trc_path, marker_name)[0:3]
        elif filter_name == 'median':
            x, y, z = smooth_marker_median(trc_path, marker_name)[0:3]
        elif filter_name == 'bessel':
            x, y, z = smooth_marker_bessel(trc_path, marker_name)[0:3]

    fig = plt.figure(figsize=(6,6))
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_zlabel('Y (m)')
    ax.set_title(f'{marker_name} 3D Trajectory' + (f" ({filter_name})" if filter_name else ""))

    # 범위 자동 설정
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(z.min(), z.max())
    ax.set_zlim(y.min(), y.max())

    # 범위 수동 설정
    #ax.set_xlim(-0.5, 1.5) #X축 범위
    #ax.set_ylim(1, -1)     #Z축 범위
    #ax.set_zlim(0, 2)      #Y축 범위

    line, = ax.plot([], [], [], linestyle='', marker='o', markersize=5)
    ax.set_xlim(x.min(), x.max()); ax.set_ylim(z.min(), z.max()); ax.set_zlim(y.min(), y.max())

    def update(frame):
        xi, yi, zi = x[frame], y[frame], z[frame]
        line.set_data([xi], [zi])
        line.set_3d_properties([yi])
        return line,

    return FuncAnimation(fig, update, frames=len(t), interval=interval, blit=True)


def animate_all_markers_3d(trc_path: str,
                           interval: int = 50,
                           show_trails: bool = False,
                           filter_name: str = None) -> FuncAnimation:
    """
    모든 마커의 3D 궤적 애니메이션
    filter_name: None | 'butterworth' | 'median' | 'bessel'
    """
    t       = extract_time_numpy(trc_path)
    markers = get_marker_names(trc_path)

    # 각 마커별 (X,Z,Y) 데이터 준비
    data_xyz = {}
    for m in markers:
        if filter_name is None:
            x, y, z = extract_marker_numpy(trc_path, m)
        else:
            if filter_name == 'butterworth':
                x, y, z = smooth_marker_butterworth(trc_path, m)[0:3]
            elif filter_name == 'median':
                x, y, z = smooth_marker_median(trc_path, m)[0:3]
            elif filter_name == 'bessel':
                x, y, z = smooth_marker_bessel(trc_path, m)[0:3]
        data_xyz[m] = (x, z, y)  # plotting order X,Z,Y

    fig = plt.figure(figsize=(7,7))
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_zlabel('Y (m)')
    ax.set_title("All Markers 3D Trajectories" + (f" ({filter_name})" if filter_name else ""))


    # 범위 수동 설정
    ax.set_xlim(0, 4)      #X축 범위
    ax.set_ylim(2.0, -2.0) #Z축 범위
    ax.set_zlim(0, 4)      #Y축 범위

    cmap = plt.get_cmap('tab20')
    lines = {}
    for i, m in enumerate(markers):
        color = cmap(i % 20)
        line, = ax.plot([], [], [], linestyle='', marker='o', markersize=4, color=color, label=m)
        lines[m] = line
    ax.legend(loc='upper right', bbox_to_anchor=(1.3,1.0), fontsize='small')

    def update_all(frame):
        artists = []
        for m in markers:
            x_plot, z_plot, y_plot = data_xyz[m]
            line = lines[m]
            if show_trails:
                line.set_data(x_plot[:frame+1], z_plot[:frame+1])
                line.set_3d_properties(y_plot[:frame+1])
            else:
                line.set_data([x_plot[frame]], [z_plot[frame]])
                line.set_3d_properties([y_plot[frame]])
            artists.append(line)
        return artists

    return FuncAnimation(fig, update_all, frames=len(t), interval=interval, blit=True)