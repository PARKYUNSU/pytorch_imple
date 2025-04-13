# Hyperbolic Vision Transformers

"Combining Improvements in Metric Learning" - 2022

ㅡ Aleksandr Ermolov, Leyla Mirvakhabova, Valentin Khrulkov, Nicu Sebe, Ivan Oseledets


[Read the Paper](https://arxiv.org/pdf/2203.10833)

---
![image](https://github.com/user-attachments/assets/a75d338f-495b-49c5-9f78-694b695b6d71)



# 1. Introduction

Metric Learning은 데이터의 표상(representation)을 학습하는 방법의 일종입니다. 핵심 목표는 Data Embedding 간의 거리를 학습하여 의미적으로 유사한 데이터가 더 가까워지도록 하는 것입니다.

즉, 두 데이터 사이의 거리, 유사도(Similarity)라는 지표를 Loss에 반영하여 의미상 유사한 데이터는 거리가 가깝게 학습하는 것이 핵심 아이디어입니다.

이러한 방법은 인간이 사물을 인식할 때, 유사한 물체를 서로 가깝다고 판단하는 과정과 유사합니다. Metric Learning은 이러한 효과로 다양한 Computer Vision에서 사용되었습니다.

- Content-based Image Retriebal (이미지 검색)
- Near-Duplicate Detection (유사 이미지 탐지)
- Face Recognition (얼굴 인식)
- Person Re-identification (사람 재식별)
- Zero-Shot
- Few-Shot

Hyperbolic Vision Transformer(HViT) 모델의 배경은 다음과 같은 선행 연구들로 만들어졌습니다.

## 1.1 Embedding Space 선택
### Euclidean Space

기존 Metric Learning에서는 유클리드 거리(Euclidean Distance) 혹은 **코사인 유사도(Cosine Similarity)** 를 활용하여 Embedding을 학습합니다.

Ex) CNN+Fully-Connected 레이어로 구성된 고차원 임베딩(VGG 4,096차원, ViT 768차원 등).

### Hyperbolic Space
**푸앙카레 볼(Poincaré Ball)** 로 대표되는 하이퍼볼릭 임베딩은 반지름이 증가할 때 지수적(exponential) 부피 성장을 보여줍니다.

트리(Tree)나 계층적(Hierarchical) 구조를 지닌 데이터를 효율적으로 표현할 수 있으며, 유클리드 공간 대비 더 낮은 차원에서도 높은 표현력을 보인다는 장점이 있습니다.

Ex) HViT 모델은 384차원이라는 비교적 낮은 임베딩 차원에서도 충분한 표현력을 확보할 수 있습니다.

<details>
<summary>Hierarchical Data/펼치기</summary>
<div markdown="1">
<img src="https://github.com/user-attachments/assets/498b0624-0e4c-4810-b95e-a2056cc99fbb" width=500>
  
Hierarchical Data(계층적 데이터)란 데이터 항목 간에 Tree나 Graph 구조처럼 상∙하위 관계 혹은 전체-부분 관계가 존재하여 여러 Level로 조직될 수 있는 데이터를 말합니다.
  즉, 단순한 Flat한 구조가 아니라 Parent-Child 또는 전체-부분 처럼 세분화된 관계가 있는 경우를 말합니다.

1. 왼쪽 그림(Image Retrieval) Whole-Fragment 관계

   Whole 이미지가 Fragment 이미지들로 세분화 되어서, 전체 큰 궁전사진에서 지붕, 옆 건물, 잔디 등 세부요소로 나뉜 것을 확인이 가능합니다.

3. 오른쪽 그림(image degradation)

   고화질 이미지를 점진적으로 Degrade 시키면, 품질이 떠어진 이미지가 여러 클래스에 걸쳐 모호하게 대응될 수 있습니다.
   
   즉, degrade 이미지가 본래 클래스뿐 아니라, 유사한 여러 클래스로 해석이 될 수 있으므로, 이런 해석이 또다른 형태의 계층적 관계를 만든다고 볼 수 있습니다.

   Hyperbolic 공간은 이러한 계층적 특성을 지수적 Volume growth을 통해 더 자연스럽게 표현이 가능하기에 Tree구조나 계층적 구조를 지닌 데이터에 적합합니다.
<img src="https://github.com/user-attachments/assets/518385f0-a204-4a9e-91e6-1ba3b4819ca6" width=300>

| Poincaré Ball

</div>
</details>

## 1.2 Encoder Model 선택
### CNN에서 Vision Transformer 선택

기존에는 CNN(합성곱 신경망)이 이미지 임베딩 추출에 주로 사용되었으나, Transformer가 강력한 백본(backbone)으로 각광 받았습니다.

Transformer는 CNN의 위치 불변성(translation equivariance) 등의 귀납적 편향이 적어, 더 광범위한 범주의 특징을 학습할 가능성이 있습니다.

그러나, 일반화를 위해 대규모 데이터(ImageNet-21k, LAION 등)나 지식 증류(distillation), 자기지도 학습(self-supervised learning) 등의 보조 전략이 요구됩니다.

### Vision Transformer(ViT)의 장점

Patch 단위로 이미지를 처리하여, CNN과 달리 국소(local) 정보뿐만 아니라 전역(global) 문맥을 더 유연하게 학습할 수 있으며, self-attention을 통해 이미지 전역의 상호작용을 모델링할 수 있습니다.

또한, 대규모 사전훈련과정 또는 distillation, self-supervised 방식을 통해 풍부한 특징학습이 가능합니다.

## 1.3 Hyperbolic Vision Transformer
Hyperbolic Embedding의 장점과 Vision Transformer의 일반화 능력으로 HViT가 만들어졌습니다.

### Embedding Space - Hyperbolic Space
- 낮은 차원(384차원)에서도 트리·계층 구조를 효과적으로 포착
- 유클리드 거리 대신 하이퍼볼릭 거리와 역삼각함수(arctanh) 등을 통해 임베딩 간 거리를 학습

### Encoder - Vision Transformer
- CNN 대비 더 일반화된 특성 추출 가능
- 대규모 사전훈련과정 또는 distillation, self-supervised 방식을 통해 풍부한 특징학습이 가능

# 2. Method
논문은 Hyperbolic Space를 이용한 Metric Learning Loss를 제안합니다.

이 방법은 하이퍼볼릭 공간의 표현력을 활용하면서도, Cross-Entropy 손실 함수의 단순함과 일반성을 유지하는 것이 핵심입니다.

이를통해, 같은 클래스 샘플(Positive)은 가가깝게 모이도록, 다른 클래스 샘픔(Negative)은 멀어지도록 학습합니다.

## 2.1. Hyperbolic Embeddings
### Poincaré Ball(푸앙카레 볼)
Hyperbolic Space를 표현하는 여러 모델이 존재하지만, HViT에서 푸앙카레 볼 모델($D^n_c, g_D$)를 사용합니다.

모델은 $n$차원의 볼 $D^n$으로 정의되며,

$$D^n = {x ∈ \mathbb{R}^n: c∥x∥^2 < 1, c \ge 0}$$


Riemannian Metric $g_D$로 구성되어 있습니다,

$$g_D = λ^2_cg_E$$

$where:$

$g_E = I_n$ : Euclidean metric tensor

$λ_c = \frac{2}{1-c∥x∥^2}$ : conformal(공형:각도를 보존하는) 인자로, 볼의 경계 근처로 가면 $λ_c$가 커져서 거리가 무한대로 확장

<img src="https://github.com/user-attachments/assets/420f7c26-4f92-4103-8aec-91ef582620d0" width=500>
<details>

<summary>Poincaré Ball/펼치기</summary>
<div markdown="1">

## Poincaré Ball

그림은 Poincaré Ball 모델이며 $∥x∥<1$ 내부 점들로 구성됩니다.

$∥x∥=1$은 Poincaré Ball모델 밖에 있으며, 이 경계는 유한한 거리에서는 닿을수 없는 구역으로 간주됩니다.

### 1. 점 x, y (파란색 X점)
  Hyperbolic Space $D^n$ 안에 임의로 위치한 두 점이며 $d_D(x, y)$ 은 하이퍼볼릭 공간내의 두 점 x, y의 거리이다.

  유클리드 공간이었으면 |x - y|로 계산했겠지만, 그림 처럼 $d_D(x, y)$ 으로 Hyperbolic Metric에 따라 거리가 달라집니다.

### 2. $z = x ⊕_1 y$
  z는 하이퍼볼릭 덧셈(Hyperbolic addition)으로 정의된 점(빨간색 X점)

  수식은 $x ⊕_c y$으로 그림에서는 Poincaré Ball의 곡률을 $c=1$로 가정해서 $⊕_1$로 표기

  $z$는 하이퍼볼릭 기하학적 규칙을 적용해 x와 y를 합쳤을 떄 얻는 위치 입니다.

### 3. $HypAve(x,y,z)$ (초록색 X점)
  x, y, z의 하이퍼볼릭 평균입니다. 하이퍼볼릭 공간에서 세 점(x, y, z)를 비율로 균형있게 연결한 점입니다.

### 4. Poincaré Ball 경계 밖
  경계 $|x|=1$에 도달하기 위해선 유클리드 상으로는 1로 움직이면 되지만, 하이퍼볼릭 공간에서는 그 거리가 무한대로 발산합니다.

## Poincaré Ball 특징
- 경계(boundary) 근처로 갈수록 거리 값이 무한대에 가까워진다

- Euclidean Space vs Hyperbolic Space
  유클리드 공간에서는 반지름 $𝑟$ 에 따른 부피가 다항식(polynomial)적으로 증가

  하이퍼볼릭 공간에서는 부피가 지수적(exponential)으로 증가

- 트리 구조(tree-like data)를 효율적으로 표현할 수 있음

</div>
</details>

### 2.1.1. Gyrovector Formalism (자이로벡터 연산)
하이퍼볼릭 공간은 벡터 공간이아니므로 벡터 연산이 아닌 자이로벡터로 연산합니다.

#### Hyperbolic Addition
$$x⊕_c y= \frac{(1+2c⟨x,y⟩ + c∥y∥^2)x + (1−c∥x∥^2)y}{1+2c⟨x,y⟩+c^2∥x∥^2∥y∥^2}$$

$where:$

$⟨x,y⟩$ : 내적

$c$ : 푸앙카레 볼 곡률 파라미터 (실제 곡률은 $-c^2$이지만 수식 편의상 $c$)

### 2.1.2. Hyperbolic Distance
푸앙카레 볼에서 두점 $x,y ∈ D^n_c$ 사이의 거리는 다음 수식으로 연산됩니다.

$$D_{hyp}(x, y) = \frac{2}{\sqrt{c}}arctanh(\sqrt{c}∥-x⊕_c y∥)$$

$Where:$

$arctanh$ : Inverse Hyperbolic Function 

$if c→0$ 이면 유클리드 거리로 수렴 → $Euclidean: \lim_{c \to 0} D_{hyp}(x, y) = 2∥x − y∥$

### 2.1.3. Euclidean ↔ Hyperbolic Mapping
Exponential Mapping: 유클리드 벡터를 푸앙카레 모델의 점으로 유입하는 함수

Logarithmic Mapping: 하이퍼볼릭 공간의 점을 다시 유클리드 벡터로 유출하는 역함수

$$exp^c_x(v) = x⊕_c (tanh(\sqrt{c}\frac{λ^c_x∥v∥}{2})\frac{v}{\sqrt{c}∥v∥}) $$


## 2.2. Pairwise Cross-Entropy Loss
### 2.2.1. Hyperbolic Distance with Cosine Similarity

한 번의 학습 스텝에서 총 N개의 서로 다른 클래스를 선택하고, 각 클래스에서 2장씩 샘플링하여 배치크기 $K=2N$ 으로 N쌍의 positive pair를 구성합니다.

논문에서는 기존 Hyperbolic Distance($D_{hyp}$ 외에도 코사인 유사도 기반의 거리를 함계 정의 합니다.

$$D_{cos}(z_i, z_j) = ∥\frac{z_i}{∥z_i∥_2} - \frac{z_j}{∥z_j∥_2}∥^2_2 = 2 - 2 \frac{⟨z_i, z_j⟩}{∥z_i∥∥z_j∥}$$

### 2.2.2. Loss Function for a positive pair

$$l_{i, j} = -log \frac{\exp(-D(z_i, z_j)/τ)}{\sum^K_{k=1. k\neq{i}}\exp(-D(z_i, z_j)/τ)}$$

$Where:$

$D$ : Hyperbolic Distance ($D_{hyp} or D_{cis}$)

$τ$ : Temperature Hyperparameter (모델의 유사도 민감도를 조절하는 역할)

- Temperature($\tau$)를 낮추면 차이를 강조하는 효과 있음. (유사도 높은 샘플은 더 가깝게, 낮은 샘플은 더 멀게)
    
    ⇒ 명확한 경계를 형성
    
- Temperature($\tau$)를 높이면 차이를 완화하는 효과 있음.
    
    ⇒ 모든 벡터들 간의 유사도를 평평하게 만드는 경향

Positive Pair 유사도를 높이는 방향으로, 다른 클래스는(Negative Pair)는 $D_{hyp}$ 또는 $D_{cos}$로 멀게 학습하는 방법을 택합니다.

Ex) Class [’침대’, ‘테이블’, ‘램프’, ‘서랍’, ‘의자’, ‘소파’]
- 각 클래스에서 2개의 샘플 추출(침대 2개,…) ⇒ **positive pair**
- 클래스 끼리는(침대 2개 ↔ 테이블 2개) ⇒ **negative pair**

## 2.3. $δ$-hyperbolicity
Embedding할 대상 Manifold가 어떤 Curvature(곡률)을 가지고 있는지 무시되지만, 실제 데이터셋 별로 곡률을 추정하는 방식은 효율적일 수 있다고 합니다.

논문에서는 학습에 사용할 데이터셋의 Hyperbolicity를 측정하는 방법과 그 측적량을 보여줍니다.

### 2.3.1. Gromov $δ$
Gromov Product

세점 $x, y, z \in X$에 대해, 거리 함수를 $d$라 할떄 수식은,

$(y, z)_x = \frac{1}{2}(d(x, y) + d(x, z) - d(y, z))$

점 $x$를 기준점(reference point)으로 $y$와 $z$사이의 관계를 $x$와의 거리를 통해 분석
  
  
### 2.3.2. $δ$

$δ$는 $(M ⊗ M) - M$ 행렬에서 가증 큰 값으로 정의됩니다.

여기서 M은 데이터 셋에서 여러 Pair wise Gromov Product를 구해서 행렬 M으로 만든 것입니다.

$$δ = max[(M ⊗ M) - M]$$

$Where:$

$⊗$ : Min-Max 행렬 곱

- $\delta$ 값을 스케일을 다시 했을 때, $0 < \delta < 1$ 이를 $\delta$-Hyperbolicity라고 부름 
- $\delta$ 값이 0에 가까울수록 데이터 구조가 더 쌍곡적(Hyperbolic)이며, 데이터 구조가 트리(Tree)와 유사하다고 판단
- 곡률은 다음과 같이 정의됨

  $c(X) = \left( \frac{0.144}{\delta} \right)^2$
  
- $\delta$가 작을수록 $\frac{1}{\delta}$ 값이 커지므로 하이퍼볼릭 반지름이 커짐, 이를 통해 Poincaré Ball에서 최적의 반지름을 계산 가능

  $\Rightarrow$ 데이터를 가장 잘 임베딩할 수 있는 공간 크기 결정

  $\Rightarrow$ Poincaré Ball에서 데이터에 내재된 기하학적 구조에 맞게 더 효과적으로 임베딩 가능


#### Dataset $δ-Hyperbolicity$

| Encoder | CUB-200 | Cars-196 | SOP   | In-Shop |
|---------|--------:|---------:|------:|--------:|
| ViT-S   | 0.280   | 0.339    | 0.271 | 0.313   |
| DeiT-S  | 0.294   | 0.343    | 0.270 | 0.323   |
| DINO    | 0.315   | 0.327    | 0.301 | 0.318   |


## 2.4. Feature Clipping
Hyperbolic Neural Network는 학습과정에서 Poincaré ball의 경계(boundary)에 밀려 기울기(gradient)가 사라지는 vanishing gradients 문제가 자주 발생합니다.

이를 방지하기 위해서 Poincaré ball 내부 점들의 Norm을 일정 수준에서 Clipping 하는 방식을 사용합니다.

표준적으로는, $r = \sqrt{\frac{1}{c}}(1-10^{-5})$으로 Norm 값을 임계값으로 사용합니다.

그러나 논문에서는 Feature Clipping으로 vanishing gradients 문제를 보안합니다.

$x^E_C = min(1, \frac{r}{∥x^E∥})⋅x^E$



$Where:$

$x^E$ : 유클리드 공간의 Feature Vector

$x^E_C$ : Clipping된 Vector

$r$ : Poincaré ball의 새로운 effective radius

## 2.5 Vision Transformers

논문에서는 **3가지 인코더 모델**을 사용합니다:

1. ViT-S
   - ViT(Base) 모델 대비 규모 축소 (헤드 수 6, Base는 12)
   - 약 2,200만 Parameter, FLOPS 8.4
   - ImageNet-21k Pretrained

2. DeiT-S
   - ViT-S 아키텍처 사용
   - ImageNet-1k로 학습
   - CNN 기반 Teacher 모델(ResNet)로부터 Distillation
   - 더 적은 데이터로도 비교적 효율적으로 학습 가능

3. DINO (Self-Supervised Learning)
   - ViT-S 모델을 라벨 없이 자기지도학습(SSL)으로 학습
   - ImageNet-1k 사용
   - 다양한 증강(랜덤 크롭, 색상 변환 등)에 대해 일관된 표현을 학습 → 이미지 검색 등 응용 시 좋은 특성 제공

#  3. Experiments
논문에서는 널리 사용되는 학습 및 평가 프로토콜을 따르며, 4개의 벤치마크 데이터셋을 사용하여 state-of-the-art 성능과 비교했습니다.

## 3.1. Dataset
실험은 다음 4개의 벤치마크 데이터셋을 사용하여 수행됨.

### 1. CUB-200-2011 (CUB)
- 총 11,788장의 이미지, 200개 조류(bird breed) 클래스 포함.
- 훈련 데이터: 100개 클래스 (5,864장 이미지)
- 테스트 데이터: 100개 클래스 (5,924장 이미지)
- 이미지 간 차이가 미세하여 세밀한 분류가 필요한 어려운 데이터셋.

### 2.  Cars-196
- 총 16,185장의 자동차 이미지, 196개 자동차 모델 클래스 포함.
- 훈련 데이터: 98개 클래스 (8,054장 이미지)
- 테스트 데이터: 98개 클래스 (8,131장 이미지)
- 자동차 모델 간 구분이 필요하므로 세밀한 특징을 학습해야 하는 데이터셋.

### 3. Stanford Online Products (SOP)
- eBay에서 크롤링한 22,634개 제품 카테고리 포함, 총 120,053장의 이미지.
- 훈련 데이터: 11,318개 클래스 (59,551장 이미지)
- 테스트 데이터: 11,316개 클래스 (60,502장 이미지)
- 제품 간 시각적 차이가 크고, 범용적인 이미지 검색(image retrieval) 문제 해결을 위한 데이터셋.

### 4. In-shop Clothes Retrieval (InShop)
- 7,986개의 의류 카테고리, 총 54,712장의 의류 이미지 포함.
- 훈련 데이터: 3,997개 클래스 (25,882장 이미지)
- 패션 이미지 검색을 위한 데이터셋.

## 3.2. Implementation Details

### Model Configuration

| Component        | Details                                      |
|-----------------|----------------------------------------------|
| **Encoder**     | ViT-S (Vision Transformer-Small) [48]        |
| **Pretraining** | ViT-S (ImageNet-21k), DeiT-S (Distillation on ImageNet-1k), DINO (Self-supervised on ImageNet-1k) |
| **Embedding Dimensions** | Encoder Output: **384**, Final Embedding: **128** |
| **Weight Initialization** | (Semi) orthogonal matrix for final projection layer |

### Embedding Space Configuration

| Type  | Projection | Curvature (c) | Temperature (τ) | Feature Clipping Radius (r) |
|-------|------------|---------------|----------------|-----------------------------|
| **Hyperbolic (Hyp-)** | Poincaré Ball | 0.1 | 0.2 | 2.3 |
| **Euclidean (Sph-)**  | Unit Hypersphere (L2-normalized) | - | 0.1 | - |

### Training Configuration

| Component | Details |
|-----------|---------|
| **Optimizer** | AdamW |
| **Learning Rate** | DINO: **1 × 10⁻⁵**, ViT-S & DeiT-S: **3 × 10⁻⁵** |
| **Weight Decay** | 0.01 |
| **Batch Size** | 900 |
| **Gradient Clipping** | Norm 3 |
| **AMP** | Enabled (O2 mode) |
| **Training Hardware** | **NVIDIA A100 GPU** |

### Dataset & Preprocessing

| Dataset | Image Resizing |
|---------|---------------|
| **CUB-200** | 256 × 256 → 224 × 224 (center crop) |
| **Others** | 224 × 224 |

| Augmentation | Details |
|-------------|---------|
| **Random Crop** | Resized to 224 × 224 using Bicubic interpolation |
| **Random Horizontal Flip** | p = 0.5 |

### Training Steps per Dataset

| Dataset | Training Steps |
|---------|---------------|
| **CUB-200** | 200 |
| **Cars-196** | 600 |
| **SOP** | 25,000 |
| **In-Shop** | 2,200 |

### Evaluation & Metrics

| Metric | Distance Measure |
|--------|-----------------|
| **Recall@K** | **Spherical Embedding (Sph-)**: Cosine Distance ($D_{cos}$) |
| **Recall@K** | **Hyperbolic Embedding (Hyp-)**: Hyperbolic Distance ($D_{hyp}$) |

| $Recall@K = \frac{쿼리당 상위 K개 결과 중 올바른 정답 수}{전체 쿼리 수}$

## 3.3. Result
### 3.3.1. Embedding Results $(Dim=128)$
![image](https://github.com/user-attachments/assets/f6183136-578e-4c1a-a448-05e69395244d)

| 128 차원 Embedding Result (Sph- : $D_{cos}$, Hyp- : $D_{hyp}$)

### 3.3.2. Encoder Results $(Dim=384)$
![image](https://github.com/user-attachments/assets/454dce4f-3390-4321-8716-88073090bf37)

| 384 차원 Encoder Result (Sph- : $D_{cos}$, Hyp- : $D_{hyp}$)

### 3.3.3. Hyperparameters on Performance

### 1. Encoder Patch Size
Patch 단위로 처리하는 ViT는 기본 패치크기가 16 x 16인데, 패치 크기를 8 x 8로 사용할 경우 약 +4.4% 성능 향상을 확인할 수 있었다.

### 2. Manifold Curvature
- 곡률 값 $c$에 따른 모델 성능 분석

  $c$값이 0.01 ~ 0.3 범위에서 성능이 안정적, 그러나 $c$ 값이 커질 수록 성능이 저하
  
### 3. Embedding Size & Batch Size
- 임베딩 차원이 작을수록 Recall@1 성능이 감소

  하지만 In-Shop 데이터셋 (클래스 개수: 3,985개)을 고려했을 때, 낮은 차원에서도 합리적인 표현력을 유지
  
- 배치 크기(Batch Size) 는 학습 과정에서 제공되는 Negative Sample의 개수 조절
    - 배치 크기 400 이상 → 성능이 안정적
    - 배치 크기 200 → 성능 다소 저하
    - 배치 크기 1600 이상 → 추가적인 성능 향상 크지 않음

<img src="https://github.com/user-attachments/assets/1416bd5d-a481-4d0c-bcab-753d1ca55405" width=400>


### 3.3.4. Poincaré Disk Visualization

UMAP(차원 축소 기법)을 사용해 2D로 변환이 가능합니다.
  
학습된 임베딩을 보면, 각 클래스가 푸앙카레 디스크의 경계로 밀려나며 잘 분리되었습니다.
  
그러나 테스트 데이터에서는 일부 샘플이 중앙으로 몰리면서 혼합되는 현상이 발견되었으나, 이 현상은 데이터의 계층적 구조(hierarchical structure) 때문으로 해석할 수 있습니다.

<img src="https://github.com/user-attachments/assets/f8a19e97-62c4-490c-8c34-bf74b19a881e" width=500>

| Poincaré Disk 임베딩 시각화


## 4. Conclusion

### 1. 본 연구의 주요 기여
- 하이퍼볼릭 거리(hyperbolic distance)를 활용한 Pairwise Cross-Entropy Loss 도입.
- Vision Transformer를 활용한 Metric Learning 프레임워크 설계.
- 여러 사전훈련 방법(Pretraining Schemes)을 비교하여 최적 조합 분석.

### 2. 실험 결과
- 각 구성 요소가 성능 향상에 중요한 역할을 함을 실험적으로 검증.
- 특히, Self-Supervised Learning(DINO)과 Metric Learning을 결합한 Hyp-DINO 모델이 가장 강력한 성능을 보임.

### 3. Limitations
- 본 연구는 컴퓨터 비전 도메인 내에서 카테고리 기반 검색(Category-Level Retrieval)에만 초점을 맞춤.
- 하이퍼볼릭 임베딩과 트랜스포머는 원래 자연어 처리(NLP)에서 제안된 개념으로 멀티모달 학습(Vision + Language)에 대한 연구로 확장 가능.
