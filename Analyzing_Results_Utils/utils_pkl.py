import pickle
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def load_pkl(file_path: str) -> Any:
    """
    .pkl 파일을 열고 파이썬 객체로 반환

    Inputs:
        file_path: 불러올 .pkl 파일 경로

    Outputs:
        pkl 파일에 저장된 파이썬 객체
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
    except pickle.UnpicklingError:
        raise ValueError(f"파일이 pickle 형식이 아닙니다: {file_path}")
    except Exception as e:
        raise RuntimeError(f"pkl 파일 로딩 중 오류 발생: {e}")


def load_frame_times(file_path: str, fps: float = 60.0) -> np.ndarray:
    """
    .pkl 파일을 로드하고, 프레임 인덱스별 시간(초) 정보를 1D numpy array로 반환

    Inputs:
        file_path: pkl 파일 경로
        fps: 원본 영상의 프레임 레이트 (default: 60.0)

    Outputs:
        numpy.ndarray of shape (n_frames,), dtype=float
    """
    data = load_pkl(file_path)
    if not hasattr(data, "__len__"):
        raise ValueError("로드된 객체에 길이 정보가 없습니다.")
    n_frames = len(data)
    # 프레임 인덱스 0,1,2,... 에 대응하는 시간 계산
    times = np.arange(n_frames, dtype=float) / fps
    return times


def extract_keypoint_xy(file_path: str, keypoint_index: int) -> np.ndarray:
    """
    .pkl 파일을 로드하고, 지정된 keypoint_index에 해당하는
    X, Y 좌표와 Confidence Score를 (3, N) 형태의 numpy array로 반환

    Inputs:
        file_path: pkl 파일 경로
        keypoint_index: 0부터 24까지의 정수 인덱스 (총 25개 관절 중 하나)

    Outputs:
        numpy.ndarray of shape (3, n_frames)
        - row 0: X 좌표
        - row 1: Y 좌표
        - row 2: Confidence Score
    # 지정된 keypoint_index에 해당하는 X, Y 좌표와 Confidence Score 추출
    one_keypoint = extract_keypoint_xy(file_path: str, keypoint_index: int)
    print("shape of one_keypoint:", one_keypoint.shape)
    """
    data = load_pkl(file_path)

    # 객체가 시퀀스인지 확인
    if not hasattr(data, "__len__"):
        raise ValueError("로드된 객체에 길이 정보가 없습니다.")
    n_frames = len(data)

    # 인덱스 유효성 검사
    if not (0 <= keypoint_index < 25):
        raise ValueError("keypoint_index는 0부터 24 사이여야 합니다.")

    xs = np.zeros(n_frames, dtype=float)
    ys = np.zeros(n_frames, dtype=float)
    cs = np.zeros(n_frames, dtype=float)

    for i, frame in enumerate(data):
        # frame이 리스트 안에 dict 형태로 들어있다고 가정
        entry = frame[0]
        keypoints = entry.get('pose_keypoints_2d')
        if keypoints is None or len(keypoints) < (keypoint_index * 3 + 2):
            raise ValueError(f"프레임 {i} 에서 keypoints 정보가 잘못되었습니다.")
        offset = keypoint_index * 3
        xs[i] = keypoints[offset]
        ys[i] = keypoints[offset + 1]
        cs[i] = keypoints[offset + 2] 

    # (3, N) 형태로 합치기
    #return np.vstack([xs, ys])
    return np.vstack([xs, ys, cs])


def animate_all_keypoints_2d(pkl_path: str,
                             interval: int = 50,
                             show_trails: bool = False) -> FuncAnimation:
    """
    .pkl 파일에서 25개 keypoint의 2D 궤적 애니메이션을 생성
    
    Inputs:
        pkl_path:      .pkl 파일 경로
        interval:      프레임 간격(ms)
        show_trails:   True면 지나온 궤적을, False면 현재 프레임 점만 표시
    
    Returns:
        matplotlib.animation.FuncAnimation
    """
    # 시간 & 프레임수
    times = load_frame_times(pkl_path, fps=60.0)
    n_frames = len(times)
    
    # 모든 keypoint 데이터 로드 (각 keypoint 당 (2, n_frames) 배열)
    n_keypoints = 25
    data_xy = {i: extract_keypoint_xy(pkl_path, i) for i in range(n_keypoints)}
    
    # 전체 범위 계산 (고정 축)
    all_x = np.hstack([data_xy[i][0] for i in range(n_keypoints)])
    all_y = np.hstack([data_xy[i][1] for i in range(n_keypoints)])
    

    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlim(0, 720)
    ax.set_ylim(1280, 0)
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_title('All 2D Keypoints Trajectories')

    # 여기서 25개가 아니라 단 한 개의 scatter 생성
    scatter = ax.scatter([], [], s=20)

    def update(frame: int):
        # 1) 모든 키포인트 좌표, 컬러, 알파 값을 배열로 준비
        offsets = np.zeros((25, 2), dtype=float)  # 25×2
        colors  = [None]*25
        alphas  = [None]*25

        for i in range(25):
            x_vals, y_vals, c_vals = data_xy[i]
            xi, yi, ci = x_vals[frame], y_vals[frame], c_vals[frame]
            offsets[i] = (xi, yi)
            colors[i]  = 'red' if ci <= 0.8 else 'black'
            alphas[i]  = min(ci, 1.0)

        # 2) 한 번에 scatter 업데이트
        scatter.set_offsets(offsets)
        scatter.set_color(colors)
        scatter.set_alpha(alphas)

        return (scatter,)

    return FuncAnimation(
        fig, update,
        frames=n_frames,
        interval=interval,
        blit=True
    )    
 

def animate_multiple_keypoints_2d(
    pkl_paths: list[str],
    fps: float = 60.0,
    interval: int = 50,
    show_trails: bool = False
) -> FuncAnimation:
    """
    여러 .pkl 파일에 대해 1×len(pkl_paths) subplot으로
    25개 keypoint의 2D 궤적 애니메이션을 생성
    (프레임 수는 가장 짧은 시퀀스에 맞춤)

    Args:
        pkl_paths:    .pkl 파일 경로 리스트
        fps:          원본 영상 프레임 레이트 (default 60)
        interval:     프레임 간격(ms)
        show_trails:  True면 궤적을, False면 현재 프레임 점만 표시

    Returns:
        FuncAnimation 객체
    """
    n_paths = len(pkl_paths)

    # 모든 파일에 대해 keypoint 데이터를 미리 로드
    data_xy_list: list[dict[int, np.ndarray]] = []
    frame_counts: list[int] = []
    for path in pkl_paths:
        data_xy = {i: extract_keypoint_xy(path, i) for i in range(25)}
        data_xy_list.append(data_xy)
        # 각 keypoint 배열의 두 번째 차원이 프레임 수
        frame_counts.append(data_xy[0].shape[1])

    n_frames = min(frame_counts)
    fig, axes = plt.subplots(1, len(pkl_paths), figsize=(6*len(pkl_paths), 6))
    axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]

    anims = []
    for ax, data_xy in zip(axes, data_xy_list):
        ax.set_xlim(0, 720)
        ax.set_ylim(1280, 0)
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        sc = ax.scatter([], [], s=20)
        def update(frame, sc=sc, data_xy=data_xy):
            offs, cols, alps = [], [], []
            for i in range(25):
                x, y, c = data_xy[i]
                xi, yi, ci = x[frame], y[frame], c[frame]
                offs.append((xi, yi))
                cols.append('red' if ci <= 0.8 else 'black')
                alps.append(min(ci, 1.0))
            sc.set_offsets(offs)
            sc.set_color(cols)
            sc.set_alpha(alps)
            return (sc,)
        anims.append(FuncAnimation(fig, update, frames=n_frames,
                                  fargs=(), interval=interval, blit=True))
    return anims[0] if len(anims)==1 else anims


def merge_confidence_arrays(
    pkl_paths: list[str],
    keypoint_index: int
) -> np.ndarray:
    """
    여러 .pkl 파일에서 지정된 keypoint_index의 confidence score만을 추출하여 1차원 numpy array로 병합(concatenate)

    Inputs:
        pkl_paths:        .pkl 파일 경로 리스트
        keypoint_index:   0부터 24까지의 keypoint 인덱스

    Outputs:
        numpy.ndarray of shape (sum_i n_frames_i,)
        - 리스트 순서대로 각 파일의 confidence 시퀀스가 이어 붙여진 1D array
    """
    all_cs = []
    for path in pkl_paths:
        # extract_keypoint_xy가 (3, N) 배열을 반환하므로, 세 번째 행(인덱스 2)만 가져옴
        _, _, cs = extract_keypoint_xy(path, keypoint_index)
        all_cs.append(cs)

    # 리스트의 모든 1D 배열을 하나로 이어 붙여 반환
    return np.concatenate(all_cs, axis=0)


def merge_all_confidence_arrays(
    pkl_paths: list[str]
) -> np.ndarray:
    """
    여러 .pkl 파일에서 0번부터 24번까지 모든 keypoint의 confidence score를
    순서대로 추출하여 하나의 1D numpy array로 병합(concatenate)

    순서:
      for each path in pkl_paths:
        for kp_idx in 0..24:
          append extract_keypoint_xy(path, kp_idx)[2]

    Inputs:
        pkl_paths: .pkl 파일 경로 리스트

    Outputs:
        numpy.ndarray of shape (sum_i (25 * n_frames_i),)
        - 리스트 순서대로, 각 파일 내 keypoint0의 모든 프레임 confidence,
          keypoint1의 모든 프레임 confidence, …, keypoint24의 모든 프레임 confidence
    """
    all_cs = []
    for path in pkl_paths:
        for kp_idx in range(25):
            # extract_keypoint_xy 반환의 3행째가 confidence 벡터
            _, _, cs = extract_keypoint_xy(path, kp_idx)
            all_cs.append(cs)
    # 1D array로 이어 붙이기
    return np.concatenate(all_cs, axis=0)


def truncate_decimals(arr: np.ndarray, decimals: int = 3) -> np.ndarray:
    """
    1D numpy array의 각 요소를 소수점 아래 `decimals` 자리까지만 남기고,
    그 이하 자릿수는 버림(truncate)

    Inputs:
        arr:       1차원 numpy array of floats
        decimals:  남길 소수점 자리 수 (기본 3)

    Outputs:
        numpy.ndarray: 소수점 아래 `decimals` 자리만 남긴 새로운 1D array
    """
    factor = 10 ** decimals
    # arr * factor 를 버림(truncate)한 뒤 다시 factor로 나눔
    return np.trunc(arr * factor) / factor


def plot_confidence_distribution(
    pkl_paths: list[str],
    decimals: int = 5
) -> plt.Axes:
    """
    주어진 .pkl 파일 리스트에서 0~24번 keypoint의 confidence score를
    모두 추출하여 소수점 아래 `decimals` 자리까지만 남긴 후(버림),
    그 값들의 빈도수를 히스토그램 형태로 Plot

    Args:
        pkl_paths: .pkl 파일 경로 리스트
        decimals: 남길 소수점 자리 수 (기본 5)

    Returns:
        matplotlib Axes 객체
    """
    # 모든 파일·모든 keypoint의 confidence를 하나로 병합
    all_conf = merge_all_confidence_arrays(pkl_paths)

    # 소수점 아래 `decimals` 자리까지만 남기고 truncate
    truncated = truncate_decimals(all_conf, decimals=decimals)

    # 고유값별 빈도 계산
    vals, counts = np.unique(truncated, return_counts=True)

    # line plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(vals, counts, linewidth=0.5, linestyle='-')
    ax.set_xlabel(f'Confidence Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Confidence Score Distribution')
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    return ax