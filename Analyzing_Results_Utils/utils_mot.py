import pandas as pd
import numpy as np
from typing import List
import matplotlib.pyplot as plt


def get_kinematics_names(mot_path: str) -> List[str]:
    """
    .mot 파일을 읽어서 'time'부터 'pro_sup_l'까지의 kinematics 데이터 Column 이름을 반환

    Inputs:
        mot_path: .mot 파일 경로

    Outputs:
        'time'부터 'pro_sup_l'까지의 Column 이름 리스트
    """
    # 헤더 끝 행 찾기
    header_end = None
    with open(mot_path, "r") as f:
        for idx, line in enumerate(f):
            if line.strip().lower() == "endheader":
                header_end = idx
                break
    if header_end is None:
        raise ValueError("mot 파일에서 'endheader'를 찾을 수 없습니다.")

    # 데이터 읽기
    df = pd.read_csv(
        mot_path,
        sep="\t",
        skiprows=header_end+1,
        engine="python"
    )

    cols = list(df.columns)
    try:
        start = cols.index("time")
        end   = cols.index("pro_sup_l")
    except ValueError as e:
        raise ValueError(f"필요한 kinematics 데이터가 없습니다: {e}")

    return cols[start:end+1]


def get_kinematics_values(mot_path: str, column_name: str) -> np.ndarray:
    """
    .mot 파일에서 특정 kinematics Column의 값을 1차원 numpy array로 반환

    Inputs:
        mot_path:     .mot 파일 경로
        column_name:  반환할 kinematics Column 이름 (예: 'pro_sup_l')

    Outputs:
        해당 Column의 값이 담긴 1차원 numpy array

    Raises:
        ValueError: mot 파일에 'endheader'가 없거나, 요청한 Column이 없을 때
    """
    # 헤더 끝 행 찾기
    header_end = None
    with open(mot_path, "r") as f:
        for idx, line in enumerate(f):
            if line.strip().lower() == "endheader":
                header_end = idx
                break
    if header_end is None:
        raise ValueError("mot 파일에서 'endheader'를 찾을 수 없습니다.")

    # 헤더 이후부터 데이터로 읽기
    df = pd.read_csv(
        mot_path,
        sep="\t",
        skiprows=header_end + 1,
        engine="python"
    )

    # 요청한 Column이 있는지 확인
    if column_name not in df.columns:
        raise ValueError(f"mot 파일에 '{column_name}' Column이 없습니다. "
                         f"사용 가능한 Column: {list(df.columns)}")

    # numpy array로 반환
    return df[column_name].values


def plot_kinematics_values(mot_path: str, column_name: str) -> None:
    """
    .mot 파일에서 'time'을 x축으로, 지정한 kinematics Column을 y축으로 하여 Plot

    Inputs:
        mot_path:     .mot 파일 경로
        column_name:  플롯할 kinematics Column 이름 (예: 'elbow_flex_r')
    """
    # 시간 정보와 원하는 컬럼 값을 가져옴.
    time = get_kinematics_values(mot_path, "time")
    values = get_kinematics_values(mot_path, column_name)

    # 배열 길이 맞춤 검증
    if time.shape != values.shape:
        raise ValueError(f"time과 {column_name} 데이터 길이가 일치하지 않습니다: "
                         f"{time.shape} vs {values.shape}")

    # plot
    plt.figure(figsize=(8, 4))
    plt.plot(time, values, linewidth=1.5)
    plt.xlabel("Time (s)")
    plt.ylabel(column_name)
    plt.title(f"{column_name} Graph")
    plt.grid(True)
    plt.tight_layout()
    plt.show()