import numpy as np
from scipy.signal import butter, filtfilt, medfilt, bessel
from utils_trc import *


def smooth_marker_butterworth(trc_path: str,
                          marker_name: str,
                          cutoff_freq: float = 2.0,
                          filter_order: int = 4) -> np.ndarray:
    """
    .trc 파일과 마커 이름을 받아서 각 축(X,Y,Z)에 Butterworth low-pass filter를 적용하고, 
    시간까지 포함한 shape (4, N) 배열로 반환

    반환값:
      row 0 : Time (N,)
      row 1 : X 축 스무딩 값 (N,)
      row 2 : Y 축 스무딩 값 (N,)
      row 3 : Z 축 스무딩 값 (N,)

    Parameters
    ----------
    trc_path : str
        .trc 파일 경로
    marker_name : str
        마커 이름 (예: 'Neck')
    cutoff_freq : float
        Low-pass Cutoff 주파수 (Hz)
    filter_order : int
        필터 차수 (기본 4)
    """
    # 시간 & 원본 좌표 추출
    time = extract_time_numpy(trc_path)                    # shape (N,)
    x, y, z = extract_marker_numpy(trc_path, marker_name)  # shape (3, N)

    # 샘플링 주파수 계산
    dt = np.mean(np.diff(time))
    fs = 1.0 / dt
    nyq = fs / 2.0
    normal_cutoff = cutoff_freq / nyq

    # Butterworth 필터 설계
    b, a = butter(filter_order, normal_cutoff, btype='low', analog=False)

    # filtfilt로 양 방향 스무딩
    x_smooth = filtfilt(b, a, x)
    y_smooth = filtfilt(b, a, y)
    z_smooth = filtfilt(b, a, z)

    # 결과 합치기
    return np.vstack([x_smooth, y_smooth, z_smooth])


def smooth_marker_median(trc_path: str,
                         marker_name: str,
                         kernel_size: int = 7) -> np.ndarray:
    """
    .trc 파일과 마커 이름을 받아서 각 축(X,Y,Z)에 Median 필터를 적용하고, 
    시간까지 포함한 shape (4, N) 배열로 반환

    반환값:
      row 0 : Time (N,)
      row 1 : X 축 스무딩 값 (N,)
      row 2 : Y 축 스무딩 값 (N,)
      row 3 : Z 축 스무딩 값 (N,)

    Parameters
    ----------
    trc_path : str
        .trc 파일 경로
    marker_name : str
        마커 이름 (예: 'Neck')
    kernel_size : int
        Median 필터 윈도우 크기 (홀수)
    """
    # 시간 & 원본 좌표 추출
    time = extract_time_numpy(trc_path)                    # shape (N,)
    x, y, z = extract_marker_numpy(trc_path, marker_name)  # shape (3, N)

    # Median 필터 적용 (홀수 커널 권장)
    x_med = medfilt(x, kernel_size=kernel_size)
    y_med = medfilt(y, kernel_size=kernel_size)
    z_med = medfilt(z, kernel_size=kernel_size)

    # 결과 합치기
    return np.vstack([x_med, y_med, z_med])


def smooth_marker_bessel(trc_path: str,
                         marker_name: str,
                         cutoff_freq: float = 6.0,
                         filter_order: int = 4) -> np.ndarray:
    """
    .trc 파일과 마커 이름을 받아서 각 축(X,Y,Z)에 Bessel 저역통과 필터를 적용하고, 
    시간까지 포함한 shape (4, N) 배열로 반환

    반환값:
      row 0 : Time (N,)
      row 1 : X 축 스무딩 값 (N,)
      row 2 : Y 축 스무딩 값 (N,)
      row 3 : Z 축 스무딩 값 (N,)

    Parameters
    ----------
    trc_path : str
        .trc 파일 경로
    marker_name : str
        마커 이름 (예: 'Neck')
    cutoff_freq : float
        Low-pass Cutoff 주파수 (Hz)
    filter_order : int
        필터 차수 (기본 4)
    """
    # 시간 & 원본 좌표 추출
    time = extract_time_numpy(trc_path)                    # shape (N,)
    x, y, z = extract_marker_numpy(trc_path, marker_name)  # shape (3, N)

    # 샘플링 주파수 계산
    dt = np.mean(np.diff(time))
    fs = 1.0 / dt
    nyq = fs / 2.0
    normal_cutoff = cutoff_freq / nyq

    # Bessel 필터 설계
    b, a = bessel(filter_order, normal_cutoff, btype='low', analog=False)

    # filtfilt로 양 방향 스무딩
    x_bes = filtfilt(b, a, x)
    y_bes = filtfilt(b, a, y)
    z_bes = filtfilt(b, a, z)

    # 결과 합치기
    return np.vstack([x_bes, y_bes, z_bes])