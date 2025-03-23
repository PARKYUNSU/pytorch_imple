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
    
    src_pts = np.float32([k_pts1[m.queryIdx].pt for m in conf]).reshape(-1, 1, 2)
    dst_pts = np.float32([k_pts2[m.trainIdx].pt for m in conf]).reshape(-1, 1, 2)
    cam4_to_cam1, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    np.save(f"{opts.homography_pth}.npy", cam4_to_cam1)

    src_pts = np.int0(src_pts).reshape(-1, 2)
    dst_pts = np.int0(dst_pts).reshape(-1, 2)

    img_matches = utilities.draw_matches(frame1, src_pts, frame2, dst_pts)
    assert cv2.imwrite("./img_with_matches.png", img_matches)

    detector = torch.hub.load("ultralytics/yolov5" , "yolov5m")

    detector.agnostic = True
    detector.classes =[0]

    num_frames1 = video1.get(cv2.CAP_PROP_FRAME_COUNT)
    num_frames2 = video2.get(cv2.CAP_PROP_FRAME_COUNT)
    num_frames = min(num_frames2, num_frames1)
    num_frames = int(num_frames)

    # Second video's frames is 17
    video2.set(cv2.CAP_PROP_FRAME_COUNT, 17)

    for idx in range(num_frames):

        frame1 = video1.read()[1]
        frame2 = video2.read()[1]

        annotation = detector([frame1, frame2])

        pred1 = annotation.xyxy[0].cpu().numpy()[:, :4]
        pred2 = annotation.xyxy[1].cpu().numpy()[:, :4]

        pred2 = utilities.apply_homography_xyxy(pred1, cam4_to_cam1)
        
        utilities.draw_bounding_boxes(frame1, pred1)
        utilities.draw_bounding_boxes(frame2, pred2)
        utilities.draw_bounding_boxes(frame2, pred2, color=(0, 0, 255))

        vis = np.concatenate([frame1, frame2], axis=1)

        cv2.namedWindow("vis", cv2.WINDOW_NORMAL)
        cv2.imshow("vis", vis)
        key = cv2.waitKey(0)

        if key == ord("q"):
            break

    video1.release()
    video2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video1", type=str, help="Path to the video cam1.mp4.")
    parser.add_argument("--video2", type=str, help="Path to the video cam4.mp4.")
    parser.add_argument(
        "--homography-pth",
        type=str,
    )
    opts = parser.parse_args()
    main(opts)