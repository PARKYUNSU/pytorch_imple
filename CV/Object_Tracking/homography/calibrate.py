import torch
import numpy as np
import cv2
import utilities

def main(opts):
    video1 = cv2.VideoCapture(opts.video1)
    if not video1.isOpened():
        raise IOError(f"Could not open video1 source {opts.video1}")

    video2 = cv2.VideoCapture(opts.video2)
    if not video2.isOpened():
        raise IOError(f"Could not open video2 source {opts.video2}")
    
    # SIFT 1000개 Keypoints 설정
    feat_detector = cv2.SIFT_create(1000)

    _, frame1 = video1.read()
    _, frame2 = video2.read()

    k_pts1, vec1 = feat_detector.detectAndCompute(frame1, None) # None 설정 시 전체이미지
    k_pts2, vec2 = feat_detector.detectAndCompute(frame2, None) # mask를 넣으면 특정 위치 지정 가능

    # keypoints matching
    match = cv2.BFMatcher()

    matches = match.knnMatch(vec1, vec2, k = 2) # k = 2 : 유클리드 거리가 가까운 두 점

    conf = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            conf.append(m)