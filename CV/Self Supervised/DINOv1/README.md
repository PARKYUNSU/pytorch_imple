# DINO V1

"Emerging Properties in Self-Supervised Vision Transformers" - 2021

ㅡ Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, Armand Joulin

[Read the Paper](https://arxiv.org/pdf/2104.14294)

---

# 1. Introduction
[Transformer](https://github.com/PARKYUNSU/pytorch_imple/tree/main/NLP/Transformer)는 시각적 인식을 위한 대안으로 Convolution Network를 대체하는 모델로 주목 받았습니다.

Computer Vision 분야에서도 대량의 데이터에 대한 사전 학습을 수행한 후 파인튜닝하는 Trnasformer에 영감을 받아 [Vision Transformer](https://github.com/PARKYUNSU/pytorch_imple/tree/main/CV/Classification/Vision_Transformer)를 만들 수 있었습니다.

기존 CNN과 경쟁력을 갖춘 Vision Transformer 였지만, 논문의 저자는 ViT 모델은 명확한 이점을 제시하지 못한다고 평가 했습니다. ViT는 계산 비용이 많이 들고, 더 많은 데이터가 필요하며, Inductive Bias가 부족하다는 한계가 있습니다.

이러한 문제점을 극복하기 위해 NLP에서 성공한 Self-Supervised Learning 방식을 ViT에 접목시키고자 하는 것이 본 논문의 취지입니다.

# 2. Approach
## 2.1. SSL(Self-Supervised Learning) with Knowledge Distillation
DINO에서는 입력 이미지 $x$에 대해 두 개 이상의 왜곡된 뷰(views)를 생성합니다. Student 네트워크 $g_{\theta_{s}}$와 Teacher 네트워크 $g_{\theta_{t}}$는 동일한 아키텍처 $g$를 공유하지만 파라미터는 공유하지 않습니다.
각 네트워크는 입력 이미지에 대해 $K$차원의 logit 벡터를 출력하고, 이를 Temperature 파라미터를 이용한 SoftMax 정규화를 통해 확률분포로 변환합니다.

Sudent 네트워크의 출력에 대해 Softmax 정규화로 얻는 확률 값

$$P_s(x)^{(i)} = \frac{exp(\frac{g_{\theta_{s}(x)^{(i)}}}{\tau_{s}})}{\sum^K_{k=1}exp(\frac{g_{\theta_{s}(x)^{(k)}}}{\tau_{s}})}\quad for i = 1, ..., K$$

$Where$

$$x$$: 이미지

$$g_{\theta_{s/t}}$$ : Student / Teacher network

$$\theta_{s/t}$$ : Student / Teacher Parameters
