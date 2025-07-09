import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


def get_marker_names(file_path):
    """
    .trc 파일의 헤더에서 'Frame'과 'Time'이 있는 줄을 찾아
    그 뒤에 오는 X/Y/Z 좌표 컬럼 이름을 기준으로 고유한 마커 이름 목록을 반환
    """
    with open(file_path, 'r') as f:
        for line in f:
            # 'Frame'과 'Time'으로 시작하는 헤더 라인 탐색
            if line.strip().startswith('Frame') and 'Time' in line:
                # 탭 구분자(.trc 표준)로 분리, 없으면 공백 분리
                headers = (line.strip().split('\t')
                           if '\t' in line
                           else line.strip().split())
                break
        else:
            raise RuntimeError("헤더 라인을 찾을 수 없습니다: 'Frame'과 'Time'이 포함된 줄이 필요합니다.")
    
    # 첫 두 컬럼(Frame, Time) 뒤의 모든 컬럼이 마커 좌표(X/Y/Z)
    coord_cols = headers[2:]
    
    # X/Y/Z 접미사를 제거해 고유 마커 이름만 추출
    markers = []
    for col in coord_cols:
        # 기존 로직 그대로
        if len(col) > 1 and col[-1] in ('X','Y','Z'):
            name = col[:-1]
        else:
            name = col
        # 빈 문자열은 건너뛰기
        if not name:
            continue
        if name not in markers:
            markers.append(name)
    return markers


def extract_time_numpy(trc_path: str) -> np.ndarray:
    """
    .trc 파일에서 Time Column의 모든 데이터를 1차원 numpy array로 추출

    반환 배열의 shape은 (N,) 이며,
      요소 : time 값
    """
    # 헤더 라인 찾기 (Frame, Time 이 있는 줄)
    with open(trc_path, 'r') as f:
        lines = f.readlines()
    header_idx = None
    for idx, line in enumerate(lines):
        if line.strip().startswith('Frame') and 'Time' in line:
            header_idx = idx
            # header_cols 는 여기선 쓰지 않지만, 일관성 위해 뽑아두면 다른 함수에도 재활용 가능
            header_cols = (line.strip().split('\t')
                           if '\t' in line
                           else line.strip().split())
            break
    if header_idx is None:
        raise RuntimeError("헤더 라인(‘Frame’과 ‘Time’ 포함)을 찾을 수 없습니다.")
    
    # 실제 데이터 시작 줄 찾기 (숫자로 시작)
    data_start = None
    for idx in range(header_idx + 1, len(lines)):
        if lines[idx].strip() and lines[idx].strip()[0].isdigit():
            data_start = idx
            break
    if data_start is None:
        raise RuntimeError("데이터 시작 줄을 찾을 수 없습니다.")
    
    # 전체 마커 리스트와 컬럼명 구성 (extract_marker_numpy 방식 그대로)
    markers = get_marker_names(trc_path)
    cols = ['Frame', 'Time'] + [f"{m}{axis}" for m in markers for axis in ('X','Y','Z')]

    # pandas로 읽어서 Time Column만 NumPy로 반환
    df = pd.read_csv(
        trc_path,
        sep='\t',
        header=header_idx,        # "Frame ... Time ... MarkerX ..." 이 줄을 헤더로 사용
        skiprows=[header_idx+1],  # 바로 아래 Units 행만 건너뛰기
        usecols=['Time'],         # Time Column 하나만 읽기
        dtype=float,
        na_values=['', 'NaN']
    )
    return df['Time'].to_numpy()


def extract_marker_numpy(trc_path: str, marker_name: str) -> np.ndarray:
    """
    .trc 파일에서 지정한 마커(marker_name)의 X, Y, Z 데이터를 numpy array로 추출

    반환 배열의 shape은 (3, N)이며,
      row 0 : X
      row 1 : Y
      row 2 : Z
    """
    # 헤더 라인 찾기 (Frame, Time 등이 있는 줄)
    with open(trc_path, 'r') as f:
        lines = f.readlines()
    header_idx = None
    for idx, line in enumerate(lines):
        if line.strip().startswith('Frame') and 'Time' in line:
            header_idx = idx
            header_cols = line.strip().split('\t') if '\t' in line else line.strip().split()
            break
    if header_idx is None:
        raise RuntimeError("헤더 라인(‘Frame’과 ‘Time’ 포함)을 찾을 수 없습니다.")
    
    # 실제 데이터가 시작되는 줄 찾기 (숫자로 시작하는 첫 줄)
    data_start = None
    for idx in range(header_idx + 1, len(lines)):
        if lines[idx].strip() and lines[idx].strip()[0].isdigit():
            data_start = idx
            break
    if data_start is None:
        raise RuntimeError("데이터 시작 줄을 찾을 수 없습니다.")

    # pandas로 데이터 읽기: header=None, skiprows 까지 수동으로 처리
    markers = get_marker_names(trc_path)
    # ['Neck','RShoulder','RElbow',...]
    header_cols = ['Frame', 'Time'] + [f"{m}{axis}"
                                       for m in markers
                                       for axis in ('X','Y','Z')]

    df = pd.read_csv(
        trc_path,
        sep='\t',
        header=None,
        names=header_cols,
        skiprows=data_start,      # data_start 이전 줄(헤더 전부) 건너뛰기
        dtype=float,
        na_values=['', 'NaN']
    )
    
    # 컬럼 이름 구성
    #time_col = 'Time'
    x_col = f'{marker_name}X'
    y_col = f'{marker_name}Y'
    z_col = f'{marker_name}Z'
    for col in (x_col, y_col, z_col):
        if col not in df.columns:
            raise KeyError(f"컬럼 '{col}' 을(를) 찾을 수 없습니다. 마커 이름을 확인해주세요.")
    
    # numpy 배열로 변환 및 쌓기
    #time_vals = df[time_col].to_numpy()
    x_vals = df[x_col].to_numpy()
    y_vals = df[y_col].to_numpy()
    z_vals = df[z_col].to_numpy()
    
    return np.vstack([x_vals, y_vals, z_vals])