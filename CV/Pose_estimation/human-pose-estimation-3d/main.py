import cv2
import torch
import numpy as np
import time
import matplotlib.pyplot as plt

from model import PoseEstimationWithMobileNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Input: 1 x 3 x 256 x 448, BGR
input_height, input_width = 256, 448
# Output: 32 x 56 (256/8, 448/8)
output_height, output_width = 32, 56
scale_factor = 8

num_features_channels = 57  # 19 keypoints × 3 D
num_heatmaps = 19          # 2D keypoint heatmap
num_pafs = 38              # keypoints

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

output_video_filename = 'annotated_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 20.0
video_out = cv2.VideoWriter(output_video_filename, fourcc, fps, (input_width, input_height))
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_resized = cv2.resize(frame, (input_width, input_height))
    
    # numpy -> torch B, C, H, W
    input_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).unsqueeze(0).float()
    input_tensor = (input_tensor - 128.0) / 255.0
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        features_out, heatmaps_out, pafs_out = model(input_tensor)
    
    if isinstance(features_out, list):
        features_tensor = features_out[0]
    else:
        features_tensor = features_out
    
    # tensor shape
    # features_out: [1, 57, 32, 56]
    # heatmaps_out: [1, 19, 32, 56]
    # pafs_out: [1, 38, 32, 56]
    features_np = features_tensor.squeeze(0).cpu().numpy()  # shape: (57, 32, 56)
    heatmaps_np = heatmaps_out.squeeze(0).cpu().numpy()    # shape: (19, 32, 56)
    # pafs_np = pafs_out.squeeze(0).cpu().numpy()

    keypoints_2d = []
    keypoints_3d = []
    for i in range(num_heatmaps):
        heatmap = heatmaps_np[i]
        # heatmap (row, col)
        row, col = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        x_2d = int(col * scale_factor)
        y_2d = int(row * scale_factor)
        keypoints_2d.append((x_2d, y_2d))

        # 3D keypoints : features 57 chanel, keypoint each 3 chanel (x, y, z)
        x3d = features_np[i * 3, row, col]
        y3d = features_np[i * 3 + 1, row, col]
        z3d = features_np[i * 3 + 2, row, col]
        keypoints_3d.append((x3d, y3d, z3d))

    # 2D keypoint visualization
    annotated_frame = frame_resized.copy()
    for kp in keypoints_2d:
        cv2.circle(annotated_frame, kp, radius=3, color=(0, 0, 255), thickness=-1)

    video_out.write(annotated_frame)
    
    print("2D Keypoints:", keypoints_2d)
    print("3D Keypoints:", keypoints_3d)
    
    # 프레임 간 간단한 딜레이 (예: 50ms)
    time.sleep(0.05)

cap.release()
video_out.release()