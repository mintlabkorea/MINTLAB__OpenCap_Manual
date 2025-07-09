from utils_pkl import *
from utils_trc import *
from visual_utils_trc import *
from utils_mot import *


if __name__ == '__main__':
    
    # 2D Keypoints Data


    # 2D Keypoints Data 파일 경로 : Videos/Cam/OutputPkl/[Trial_name]_keypoints.pkl
    keypoints_path_cam0 = 'Videos/Cam0/OutputPkl/squat_keypoints.pkl' # <--[Trial_name]_keypoints.pkl 형식 잘 넣기
    keypoints_path_cam1 = 'Videos/Cam1/OutputPkl/squat_keypoints.pkl' # <--[Trial_name]_keypoints.pkl 형식 잘 넣기
    keypoints_paths = [keypoints_path_cam0, keypoints_path_cam1]


    # .pkl 파일을 열고 파이썬 객체(List)로 반환
    python_object_pkl = load_pkl(keypoints_path_cam0)
    print("Type of python_object_pkl:", type(python_object_pkl))


    # .pkl 파일을 로드하고, 프레임 인덱스별 시간(초) 정보를 1D numpy array로 반환
    frame_times = load_frame_times(keypoints_path_cam0, fps = 60.0)
    print("shape of frame_times:", frame_times.shape)


    # 지정된 keypoint_index에 해당하는 X, Y 좌표와 Confidence Score 추출
    # keypoint_index: 0부터 24까지의 정수 인덱스 (총 25개 관절 중 하나)
    one_keypoint = extract_keypoint_xy(keypoints_path_cam0, keypoint_index=10)
    print("shape of one_keypoint:", one_keypoint.shape)


    # 25개 keypoint의 2D 궤적 애니메이션 생성
    # 투명도 : Confidence Score
    # Red Points : Confidence Score < 0.8
    ani0 = animate_all_keypoints_2d(
        pkl_path=keypoints_path_cam0,
        interval=16.667,
        show_trails=False
    )
    

    # 여러 .pkl 파일에 대해 subplot으로 25개 keypoint의 2D 궤적 애니메이션 생성
    # 투명도 : Confidence Score
    # Red Points : Confidence Score < 0.8
    ani1 = animate_multiple_keypoints_2d(
        pkl_paths=keypoints_paths, #path를 담은 List
        fps=60,
        interval=20,
        show_trails=False
    )


    # Confidence Score 분포 시각화
    '''
    주어진 .pkl 파일 리스트에서 0~24번 keypoint의 confidence score를
    모두 추출하여 소수점 아래 `decimals` 자리까지만 남긴 후(버림),
    그 값들의 빈도수를 히스토그램 형태로 Plot

    Confidence Score 분포가 1에 몰려있으면,
    OpenCap으로 추출한 데이터가 신뢰할 수 있는 데이터임을 알 수 있음
    '''
    ax = plot_confidence_distribution(keypoints_paths, decimals=3)
    
    
    plt.show() # <-- 없으면 Plot이 불가능
    
    
    #-----------------------------------------------------------------------------------------------------
    
    # 3D Asanatomical Markers Data
    '''
    
    # 3D Asanatomical Markers Data 파일 경로 : MarkerData/[Trial_name].trc
    trc_path = 'MarkerData/squat.trc' # <--[Trial_name].trc 형식 잘 넣기
    marker = 'LKnee'
    

    # .trc 파일에서 3D Asanatomical Markers의 이름을 추출
    markers = get_marker_names(trc_path)
    print(markers)  # ['Neck', 'RShoulder', 'RElbow', ...]


    # .trc 파일에서 Time Column의 모든 데이터를 1차원 numpy array로 추출
    time = extract_time_numpy(trc_path)
    print(time.shape) # Ex : (721,)
    

    # .trc 파일에서 지정한 Marker의 X, Y, Z 데이터를 numpy array로 추출
    x, y, z = extract_marker_numpy(trc_path, "LKnee")
    print(x.shape, y.shape, z.shape)  # Ex : (721,) (721,) (721,)
    

    # .trc 파일과 Marker 이름, filter_name을 받아 시간–X, 시간–Y, 시간–Z 궤적을 3개의 subplot으로 한 번에 출력
    # filter name: None | 'butterworth' | 'median' | 'bessel'
    # 원본 데이터 플롯
    plot_marker_xyz(trc_path, marker, filter_name=None)
    # Bessel Filter 적용 후 Plot
    plot_marker_xyz(trc_path, marker, filter_name="bessel")
    # Butterworth Filter 적용 후 Plot
    plot_marker_xyz(trc_path, marker, filter_name="butterworth")


    # 단일 Marker의 3D 궤적 애니메이션
    # filter name: None | 'butterworth' | 'median' | 'bessel'
    ani = animate_marker_3d(trc_path, marker, interval=50 ,filter_name="median")


    # 모든 Marker의 3D 궤적 애니메이션
    # filter name: None | 'butterworth' | 'median' | 'bessel'
    ani_all = animate_all_markers_3d(trc_path, interval=50, filter_name="bessel")

    
    plt.show() # <-- 없으면 Plot이 불가능
    
    '''
    #-----------------------------------------------------------------------------------------------------
    

    # Kinematics Data
    '''
    
    # Kinematics Data 파일 경로 : OpenSimData/Kinematics/[Trial_name].mot
    mot_file = "OpenSimData/Kinematics/squat.mot" # <--[Trial_name].mot 형식 잘 넣기
    column_name = "elbow_flex_r"
    

    # .mot 파일에 들어있는 kinematics 데이터 종류 출력하기
    kinematics_names = get_kinematics_names(mot_file)
    print(kinematics_names)

    
    # .mot 파일에서 특정 kinematics Column의 값을 1차원 numpy array로 추출
    kinematics_values = get_kinematics_values(mot_file, column_name)
    print("shape of kinematics_values:", kinematics_values.shape)
    

    # .mot 파일에서 'time'을 x축으로, 지정한 kinematics Column을 y축으로 하여 Plot
    plot_kinematics_values(mot_file, column_name)


    plt.show() # <-- 없으면 Plot이 불가능
    
    '''