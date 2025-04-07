import cv2
import torch
import numpy as np
import time
import math
import matplotlib.pyplot as plt  # 필요시 사용
from google.colab import files

from model import PoseEstimationWithMobileNet

body_edges = np.array(
    [
        [0, 1],      # neck - nose
        [1, 16], [16, 18],  # nose - left eye - left ear
        [1, 15], [15, 17],  # nose - right eye - right ear
        [0, 3], [3, 4], [4, 5],   # neck - left shoulder - left elbow - left wrist
        [0, 9], [9, 10], [10, 11],  # neck - right shoulder - right elbow - right wrist
        [0, 6], [6, 7], [7, 8],     # neck - left hip - left knee - left ankle
        [0, 12], [12, 13], [13, 14]  # neck - right hip - right knee - right ankle
    ]
)

def draw_poses(img, poses_2d):
    """
    입력 이미지에 2D 포즈를 그립니다.
    poses_2d: 각 포즈는 [x1, y1, conf1, x2, y2, conf2, ..., overall_confidence] 형식의 1차원 배열
    """
    for pose in poses_2d:
        # 마지막 요소는 전체 신뢰도이므로 제외하고, (num_keypoints, 3) 배열로 변환
        keypoints = np.array(pose[:-1]).reshape((-1, 3)).transpose()  # shape: (3, num_keypoints)
        found = keypoints[2] > 0  # 신뢰도가 양수인 관절만 표시
        
        # 관절 연결선 그리기
        for edge in body_edges:
            if found[edge[0]] and found[edge[1]]:
                pt1 = tuple(keypoints[0:2, edge[0]].astype(np.int32))
                pt2 = tuple(keypoints[0:2, edge[1]].astype(np.int32))
                cv2.line(img, pt1, pt2, (255, 255, 0), 2, cv2.LINE_AA)
        # 관절 점 그리기
        for i in range(keypoints.shape[1]):
            if keypoints[2, i] > 0:
                pt = tuple(keypoints[0:2, i].astype(np.int32))
                cv2.circle(img, pt, 3, (0, 255, 255), -1, cv2.LINE_AA)

class Plotter3d:
    """
    3D 포즈(관절의 x, y, z 좌표)를 2D 평면에 투영하여 시각화하는 클래스입니다.
    """
    # 3D 스켈레톤 연결선 (관절 인덱스; 19 관절 기준)
    SKELETON_EDGES = np.array([
        [11, 10], [10, 9], [9, 0],
        [0, 3], [3, 4], [4, 5],
        [0, 6], [6, 7], [7, 8],
        [0, 12], [12, 13], [13, 14],
        [0, 1], [1, 15], [15, 16],
        [1, 17], [17, 18]
    ])

    def __init__(self, canvas_size, origin=(0.5, 0.5), scale=1):
        """
        Parameters:
            canvas_size: (높이, 너비) 튜플 – 캔버스 크기.
            origin: 캔버스 내 투영 원점 (정규화된 값, 예: (0.5, 0.5)).
            scale: 투영된 2D 좌표의 스케일 조정.
        """
        self.origin = np.array([origin[1] * canvas_size[1], origin[0] * canvas_size[0]], dtype=np.float32)
        self.scale = np.float32(scale)
        self.theta = 0  # 회전 각 (수평)
        self.phi = 0    # 회전 각 (수직)
        # 간단한 그리드(axes) 생성 – 투영 시 참고
        axis_length = 100
        axes = [
            np.array([[-axis_length/2, -axis_length/2, 0],
                      [axis_length/2, -axis_length/2, 0]], dtype=np.float32),
            np.array([[-axis_length/2, -axis_length/2, 0],
                      [-axis_length/2, axis_length/2, 0]], dtype=np.float32),
            np.array([[-axis_length/2, -axis_length/2, 0],
                      [-axis_length/2, -axis_length/2, axis_length]], dtype=np.float32)
        ]
        self.axes = np.array(axes)

    def plot(self, img, vertices, edges):
        """
        3D 포즈를 2D 평면에 투영하여 캔버스(img)에 그립니다.
        
        Parameters:
            img: 캔버스 (numpy 배열).
            vertices: (N, 3) 크기의 3D 좌표 배열.
            edges: (M, 2) 배열, 각 행은 vertices의 인덱스 연결선.
        """
        img.fill(0)
        R = self._get_rotation(self.theta, self.phi)
        self._draw_axes(img, R)
        if len(edges) != 0:
            self._plot_edges(img, vertices, edges, R)

    def _draw_axes(self, img, R):
        axes_2d = np.dot(self.axes, R)
        axes_2d = axes_2d * self.scale + self.origin
        for axe in axes_2d:
            axe = axe.astype(int)
            cv2.line(img, tuple(axe[0]), tuple(axe[1]), (128, 128, 128), 1, cv2.LINE_AA)

    def _plot_edges(self, img, vertices, edges, R):
        vertices_2d = np.dot(vertices, R)
        vertices_2d = vertices_2d * self.scale + self.origin
        for edge in edges:
            pt1 = tuple(vertices_2d[edge[0]].astype(int))
            pt2 = tuple(vertices_2d[edge[1]].astype(int))
            cv2.line(img, pt1, pt2, (255, 255, 255), 2, cv2.LINE_AA)

    def _get_rotation(self, theta, phi):
        sin, cos = math.sin, math.cos
        # 단순 투영 회전 매트릭스 (2x3 투영)
        return np.array([
            [ cos(theta), sin(theta)*sin(phi) ],
            [-sin(theta), cos(theta)*sin(phi) ],
            [ 0,         -cos(phi)         ]
        ], dtype=np.float32)[:2]  # 2행만 사용하여 2D 투영

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 모델 입력/출력 크기 등
input_height, input_width = 256, 448  # BGR 영상 크기
output_height, output_width = 32, 56   # 모델 출력 해상도 (256/8, 448/8)
scale_factor = 8

num_features_channels = 57  # 19 keypoints × 3 D
num_heatmaps = 19           # 2D keypoint heatmap
num_pafs = 38

num_refinement_stages = 1
num_channels = 128

model = PoseEstimationWithMobileNet(
    num_refinement_stages=num_refinement_stages,
    num_channels=num_channels,
    num_heatmaps=num_heatmaps,
    num_pafs=num_pafs,
    is_convertible_by_mo=True
)
model.to(device)

pth_path = '/content/pytorch_imple/CV/Pose_estimation/human-pose-estimation-3d/human-pose-estimation-3d-0001.pth'
state_dict = torch.load(pth_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()

video_path = '/content/pytorch_imple/CV/Pose_estimation/human-pose-estimation-3d/input_video.mp4'
cap = cv2.VideoCapture(video_path)

# 결과를 저장할 비디오 파일 (2D와 3D 영상을 좌우로 결합)
output_video_filename = 'annotated_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 20.0
# 최종 결합된 영상 크기는 256 x (448*2) = 256 x 896
video_out = cv2.VideoWriter(output_video_filename, fourcc, fps, (input_width * 2, input_height))

# Plotter3d 인스턴스 (3D 시각화용 캔버스 크기를 256x448로 설정)
plotter_canvas_size = (input_height, input_width)  # (256, 448)
plotter = Plotter3d(plotter_canvas_size, origin=(0.5, 0.5), scale=1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_resized = cv2.resize(frame, (input_width, input_height))
    
    # numpy -> torch 텐서 (B, C, H, W)
    input_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).unsqueeze(0).float()
    input_tensor = (input_tensor - 128.0) / 255.0
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        features_out, heatmaps_out, pafs_out = model(input_tensor)
    
    # features_out가 리스트인 경우 첫 번째 요소 사용
    if isinstance(features_out, list):
        features_tensor = features_out[0]
    else:
        features_tensor = features_out
    
    features_np = features_tensor.squeeze(0).cpu().numpy()  # (57, 32, 56)
    heatmaps_np = heatmaps_out.squeeze(0).cpu().numpy()       # (19, 32, 56)

    keypoints_2d = []
    keypoints_3d = []
    for i in range(num_heatmaps):
        heatmap = heatmaps_np[i]
        row, col = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        x_2d = int(col * scale_factor)
        y_2d = int(row * scale_factor)
        keypoints_2d.append((x_2d, y_2d))
        x3d = features_np[i * 3, row, col]
        y3d = features_np[i * 3 + 1, row, col]
        z3d = features_np[i * 3 + 2, row, col]
        keypoints_3d.append((x3d, y3d, z3d))

    # 2D annotated image: 원본 영상 위에 원(또는 draw_poses를 이용)
    annotated_2d = frame_resized.copy()
    # 여기서는 간단히 circle로 표시 (또는 draw_poses 함수를 사용)
    for kp in keypoints_2d:
        cv2.circle(annotated_2d, kp, radius=3, color=(0, 0, 255), thickness=-1)

    # 2D 포즈 배열 생성 (draw_poses 함수에서 사용할 형태: [x,y,conf] 반복 후 overall_confidence)
    # 여기서는 모든 관절이 발견되었다고 가정하고 conf=1, overall_conf=1
    pose_2d_arr = []
    for kp in keypoints_2d:
        pose_2d_arr.extend([kp[0], kp[1], 1.0])
    pose_2d_arr.append(1.0)  # overall confidence
    poses_2d_list = [pose_2d_arr]
    # 별도의 빈 이미지에 2D 포즈 그리기 (원한다면)
    img_2d = annotated_2d.copy()
    draw_poses(img_2d, poses_2d_list)

    # 3D annotated image: Plotter3d를 이용해 3D keypoints 시각화
    # keypoints_3d를 (N,3) 배열로 구성 (여기서는 19 관절)
    vertices = np.array(keypoints_3d, dtype=np.float32)
    img_3d = np.zeros((plotter_canvas_size[0], plotter_canvas_size[1], 3), dtype=np.uint8)
    plotter.plot(img_3d, vertices, Plotter3d.SKELETON_EDGES)

    # 좌우 결합: 왼쪽은 2D annotated, 오른쪽은 3D annotated
    combined_frame = np.hstack((img_2d, img_3d))
    
    video_out.write(combined_frame)
    
    print("2D Keypoints:", keypoints_2d)
    print("3D Keypoints:", keypoints_3d)
    time.sleep(0.05)

cap.release()
video_out.release()

print("Annotated video saved as:", output_video_filename)