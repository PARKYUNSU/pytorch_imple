# DINO V1

"Emerging Properties in Self-Supervised Vision Transformers" - 2021

ㅡ Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, Armand Joulin

[Read the Paper](https://arxiv.org/pdf/2104.14294)

---

# Introduction
[Transformer](https://github.com/PARKYUNSU/pytorch_imple/tree/main/NLP/Transformer)는 시각적 인식을 위한 대안으로 Convolution Network를 대체하는 모델로 주목 받았습니다.

Computer Vision 분야에서도 대량의 데이터에 대한 사전 학습을 수행한 후 파인튜닝하는 Trnasformer에 영감을 받아 [Vision Transformer](https://github.com/PARKYUNSU/pytorch_imple/tree/main/CV/Classification/Vision_Transformer)를 만들 수 있었습니다.

기존 CNN과 경쟁력을 갖춘 Vision Transformer 였지만, 논문의 저자는 ViT 모델은 명확한 이점을 제시하지 못한다고 평가 했습니다. ViT는 계산 비용이 많이 들고, 더 많은 데이터가 필요하며, Inductive Bias가 부족하다는 한계가 있습니다.

