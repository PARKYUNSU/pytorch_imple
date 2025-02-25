import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import umap
from PIL import Image
import numpy as np
from tap import Tap

# Config 클래스
class Config(Tap):
    dataset_path: str = "/home/work/ModuInterior/dataset/restructured"  # 데이터셋 경로
    model_name: str = "dino_vits16"  # 모델 이름
    img_size: int = 50  # 시각화를 위한 이미지 크기
    batch_size: int = 64  # 배치 크기

# 모델 로드 함수
def load_hvt_model(cfg):
    model = torch.hub.load('facebookresearch/dino:main', cfg.model_name, pretrained=True)
    model.head = torch.nn.Identity()  # 헤드 제거
    return model

# 메인 실행 함수
def main():
    # Config 설정
    cfg = Config().parse_args()

    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 이미지 전처리 및 데이터 로드
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ImageFolder(root=cfg.dataset_path, transform=transform)
    data_loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)

    # 모델 로드
    model = load_hvt_model(cfg)
    model = model.to(device)
    model.eval()

    # 특징 추출
    embeddings = []
    labels = []
    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            features = model(images).cpu().numpy()
            embeddings.append(features)
            labels.extend(targets.numpy())

    embeddings = np.vstack(embeddings)
    labels = np.array(labels)

    # UMAP을 사용하여 하이퍼볼릭 차원 축소
    mapper = umap.UMAP(output_metric="hyperboloid", random_state=1337)
    path2d = mapper.fit_transform(embeddings)

    # Poincaré Ball 변환
    x, y = path2d[:, 0], path2d[:, 1]
    z = (1 + x ** 2 + y ** 2) ** 0.5
    disk_x = x / (1 + z)
    disk_y = y / (1 + z)

    # 각 클래스별로 시각화
    class_names = dataset.classes
    for class_idx, class_name in enumerate(class_names):
        class_mask = labels == class_idx
        class_disk_x = disk_x[class_mask]
        class_disk_y = disk_y[class_mask]

        # Poincaré Ball 시각화
        fig = plt.figure(figsize=(10, 10), clear=True)
        ax = fig.add_subplot(111)
        scatter = ax.scatter(class_disk_x, class_disk_y, c=[class_idx]*len(class_disk_x), alpha=.75, s=7, cmap="tab10")

        # 외곽 원 추가
        boundary = plt.Circle((0, 0), 1, fc="none", ec="black", linewidth=2)
        ax.add_patch(boundary)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect('equal', adjustable='datalim')
        ax.axis("off")
        plt.colorbar(scatter, ax=ax, label="Class")
        plt.title(f"Class: {class_name}")

        # 결과 저장
        output_path = f"{class_name}_poincare_visualization2.jpg"
        plt.savefig(output_path, format="jpg", dpi=300)
        plt.close()
        print(f"시각화 결과가 {output_path}에 저장되었습니다.")

if __name__ == "__main__":
    main()
