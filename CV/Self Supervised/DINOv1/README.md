# DINO V1

"Emerging Properties in Self-Supervised Vision Transformers" - 2021

ㅡ Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, Armand Joulin

[Read the Paper](https://arxiv.org/pdf/2104.14294)

---

# 1. Introduction
[Transformer](https://github.com/PARKYUNSU/pytorch_imple/tree/main/NLP/Transformer)는 시각적 인식을 위한 대안으로 Convolution Network를 대체하는 모델로 주목 받았습니다.

Computer Vision 분야에서도 대량의 데이터에 대한 사전 학습을 수행한 후 파인튜닝하는 Trnasformer에 영감을 받아 [Vision Transformer](https://github.com/PARKYUNSU/pytorch_imple/tree/main/CV/Classification/Vision_Transformer)를 만들 수 있었습니다.

기존 CNN과 견주어 경쟁력을 갖춘 Vision Transformer 였지만, 논문의 저자는 ViT 모델은 명확한 이점을 제시하지 못한다고 평가 했습니다. ViT는 계산 비용이 많이 들고, 더 많은 데이터가 필요하며, Inductive Bias가 부족하다는 한계가 있습니다.

이러한 문제점을 극복하기 위해 NLP에서 성공한 Self-Supervised Learning 방식을 ViT에 접목시키고자 하는 것이 본 논문의 취지입니다.

# 2. Approach
## 2.1. SSL(Self-Supervised Learning) with Knowledge Distillation
DINO에서는 입력 이미지 $x$에 대해 두 개 이상의 왜곡된 뷰(views)를 생성합니다. Student 네트워크 $g_{\theta_{s}}$와 Teacher 네트워크 $g_{\theta_{t}}$는 동일한 아키텍처 $g$를 공유하지만 파라미터는 공유하지 않습니다.
각 네트워크는 입력 이미지에 대해 $K$차원의 logit 벡터를 출력하고, 이를 Temperature 파라미터를 이용한 SoftMax 정규화를 통해 확률분포로 변환합니다.

#### Sudent 네트워크의 출력 확률 분포

$$P_s(x)^{(i)} = \frac{exp(\frac{g_{\theta_{s}(x)^{(i)}}}{\tau_{s}})}{\sum^K_{k=1}exp(\frac{g_{\theta_{s}(x)^{(k)}}}{\tau_{s}})}\quad for\quad i = 1, ..., K$$

$Where$

$$x$$: Input Image

$g_{\theta_s}(x)^{(i)}$ : Student Network의 $i$ 번째 logit 값

$\tau_{s}$ : Student Temperature로 SoftMax 정규화 시 사용하며 확률 분포의 Sharpness를 조절  (여기서 $\tau_{s} > 0$)

#### Teacher 네트워크의 출력 확츌 분포

$$P_t(x)^{(i)} = \frac{exp(\frac{(g_{\theta_{t}}(x)^{(i)} - c^{(i)})}{\tau_t})}{\sum^K_{k=1}exp( \frac{g_{\theta_t}(x)^{(k)} - c^{(k)}}{\tau_t})}$$

$Where$

$g_{\theta_{t}}(x)^{(i)}$ : Teacher Network의 $i$ 번째 logit 값

$c^{(i)}$ : $K$차원 Center 벡터 $c$에서 $i$ 번째 차원 값

$$\tau_{t}$$ : Teacher Temperature로 SoftMax 정규화 시 사용하며 확률 분포의 Sharpness를 조절 (여기서 $\tau_{t} > 0$)


## 2.2 Knowledge Distillation
Knowledge Distillation는 Student Network가 Teacher Network의 출력을 모방하도록 학습하는 방식입니다. 고정된 Teacher Network $g_{\theta_t}$가 주어졌다고 가정할 때, Student Network의 파라미터 $\theta_s$는 손실함수를 최소화하도록 학습합니다.

$$
\min_{\theta_s} H\bigl(P_t(x), P_s(x)\bigr), \quad \text{with } H(a,b) = - \sum_{i=1}^{K} a^{(i)} \log b^{(i)}.
$$


그러나 DINO에서는 Teacher Network가 고정되어 있지 않고,Student Network 파라미터에 대한 지수 이동 평균(Exponential Moving Average, EMA)으로 동정 업데이트 됩니다.

## 2.3. Multi-Crop
DINO는 단순히 두 개의 변형 뷰만 사용하는 것이 아니라, 이미지 $x$ 에 Multi-Crop Augmentation을 해서 Augmented View들을 얻고 논문에서는 V리고 칭합니다.

V를 구성하는 이미지 중 Global View기 적용된 이미지는 $x^g_1, X^g_2$으로 표현되고 Local View는 $x^{'}$ 으로 표현되며, Student Network에 입력됩니다.

- Global View : 원본 이미지의 넓은 영역(예: 50% 이상)을 포함하는 해상도 224×224 크기의 (Crop된) 뷰 2개
- Local View :  원본 이미지의 작은 영역(예: 50% 미만)을 포함하는 해상도 96×96 크기의 여러 (Crop된) 뷰

모든 뷰는 Student Network에 전달되고, 오직 Global View만 Teacher Network에 전달됩니다. 이를 통해 local-to-global 대응을 유도합니다. 이때 최소화하는 손실 함수는 다음과 같이 확장됩니다.

$$
\min_{\theta_s} \sum_{x \in \{ x_g^1, x_g^2 \}} \sum_{\substack{x' \in V \\ x' \neq x}} H\bigl(P_t(x), P_s(x')\bigr),
$$

두 네트워크는 동일한 아키텍처 $g$를 사용하지만, 각각의 파라미터 $\theta_s$와 $\theta_t$를 가집니다. Student Network의 파라미터 $\theta_s$는 확률적 경사 하강법(SGD) 등을 사용하여 학습됩니다.


## DINO Diagram
<img src="https://github.com/user-attachments/assets/8e04324f-01e5-4523-bf41-46f47965ecb3" width=400>

### 입력 이미지 $x$
원본 이미지 $x$ 로부터 두 개의 서로 다른 랜덤 변환(augmentation) 뷰 $x_1$과 $x_2$를 생성합니다.

---

### Studenet Network $g_{\theta_s}$
- $x_1$을 입력으로 받아 logit을 계산한 뒤, softmax 정규화를 거쳐 확률 분포 $p_1$을 출력합니다.
- Student Network의 파라미터는 실제로 학습(역전파)되는 주체입니다.

---

### Teacher Network $g_{\theta_t}$
- $x_2$를 입력으로 받아 logit을 계산하고, centering 및 softmax 정규화를 거쳐 확률 분포 $p_2$를 출력합니다.
- Teacher Network는 지수 이동 평균(EMA) 방식으로 Student Network 파라미터 $\theta_s$를 추적합니다.  
  - 즉, 역전파로 직접 업데이트되지 않고, Student Network의 최신 파라미터가 조금씩 반영되어 보다 안정적인 타깃을 제공합니다.  
- stop-gradient (sg) Teacher Network의 출력에 대해 기울기가 역전파되지 않음을 의미합니다. (Teacher Network는 target 역할)

---

### Centering

- Teacher Network의 출력에서 각 차원의 평균값(center)을 빼 주는 연산으로, 한 차원이 지배적으로 커지는 collapse를 방지하고 모델이 더 다양한 특징을 학습하도록 유도합니다.

---

### Softmax 정규화

- Student와 Teacher Network 모두, Temperature 파라미터 $\tau_s, \tau_t$를 이용해 logit을 softmax로 변환합니다.
  - Teacher는 centering 과정을 거친 후 softmax를 적용하여 $p_2$를 얻고,
  - Student은 단순히 softmax를 적용하여 $p_1$을 얻습니다.

---

### 손실 함수 $- p_2 \log p_1$

- Studnet 출력 $p_1$이 Teacher 출력 $p_2$를 모방하도록 Cross entropy loss을 계산합니다.
  
$$
\text{loss} 
= - \sum_i p_2^{(i)} \log p_1^{(i)}
\quad 
(\text{간단히 } - p_2 \log p_1 \text{ 로 표현})
$$

---

### 지수 이동 평균 (EMA)

- Teacher Network $\theta_t$는 매 스텝에서 Student Network $\theta_s$를 모멘텀 $lambda$로 반영합니다.

$
\theta_t \;\leftarrow\; \lambda\,\theta_t + (1 - \lambda)\,\theta_s
$
