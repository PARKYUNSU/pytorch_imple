# VITS

"Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech" - 2021

ㅡ Jaehyeon Kim, Jungil Kong, Juhee Son

[Read the Paper](https://arxiv.org/pdf/2106.06103)

---

# 1. Introduction

VITS

기존 TTS 모델의 한계

Two-Stage Text-to-Speech System은 음성을 생성하기 위해 두 개의 주요 단계를 거치는 TTS 방식입니다. 이름 처럼 두가지 단계로 작동합니다.

1단계 : 텍스트를 중간 표현으로 변환

-   텍스트 입력을 받아서 이를 중간 표현으로 변환
-   일반적으로 **mel-spectrogram** 과 같은 시간-주파수 기반 표현이 생성
-   이 과정을 통해 주로 텍스트를 음소(phoneme)로 변환 후 음성의 운율(prosody), 억양(intonation), 강세(stress) 등을 모델링

주로 사용되는 모델로

Tacotron, Tacotron2, Glow-TTS 등을 사용합니다.

2단계 : 중간 표현을 음성 파형으로 변환

-   생성된 **mel-spectrogram** 을 **raw waveform**으로 변환
-   이 과정은 고주파 정보와 음향적 세부 사항을 추가하여 최종 음성을 생성하는데 초점

주로 사용되는 모델로 Vocoder가 이 단계에서 사용되며

WaveNet, WaveGlow, HiFi-GAN 등이 대표적인 보코더 입니다.

Two-Stage Text-to-Speech System 특징

장점

-   각 단계가 독립적으로 설계 및 최적화 및 모듈화 가능
-   첫 번째 단계에서 다양한 음성 특성을 학습할 수 있고, 두 번째 단계에서 고품질 음성 생성 가능

단점

-   후속단계 모델의 의존성 : **mel-spectrogram**에 의존하므로, 자연스러운 표현의 손실 가능성
    -   이를 해결하기 위해 재학습 및 파인튜닝을 해야 하므로 훈련에 추가적인 비용 및 시간이 증가
-   두 가지의 단계가 별도로 훈련되야 하므로 훈련 과정이 복잡하고 시간이 많이 소모됨
    -   **mel-spectrogram**을 생성하는 모델 훈련
    -   음성을 생성하는 보코더 훈련
-   실시간 합성 속도가 느릴 수 있음

### VITS

본 논문에서는 기존 Two-Stage Text-to-Speech System 보다 더 자연스러운 음성을 생성할 수 있는 병렬 엔드-투-엔스 TTS 방법을 제안합니다.

VAE(Variational Autoencoder)를 사용하여 TTS 시스템의 두 모듈을 잠재변수(Latent Variables)로 연결함으로 효율적인 엔드-투-엔드 학습을 가능하게 합니다.

Method

제안된 방법은 3가지 섹션으로 다루어집니다.

VAE formulation, alignment estimation, adversarial training

1. Variational Inference

    VITS는 조건부 VAE로 표현되며, 데이터의 다루기힘든 marginal log-likelihood을 최대화하는 ELBO(Evidnce Lower Bound)을 목표로 합니다.

    Marginal Likelihood

    관측 데이터 $X$가 주어졌을떄 모델의 전체 파라미터 $θ$ 가 데이터를 얼마나 잘 설명하는지 나타내는 확률

    $P(X∣α)=∫P(X∣θ)P(θ∣α)dθ$

    $Where:$

    - $P(X∣α)$: Marginal Likelihood (데이터 $X$에 대한 모델의 확률)
    - $P(X∣θ)$: 데이터 $X$가 파라미터 $θ$일 때의 Likelihood
    - $P(θ∣α)$: 파라미터 $θ$에 대한 사전 분포(Prior Distribution)
    - $∫dθ$: $θ$를 적분하여 제거(marginalize)함으로써 전체 모델 확률 계산

    **Marginalize :** 특정 변수를 제거하고 나머지 변수만으로 분포를 계산하는 과정, $θ$ 는 관찰되지 않은 잠재 변수이기 때문에 $θ$를 적분해 데이터 $X$의 확률을 계산

    Marginal Likelihood 최대화하는 이유는, 이 확률이 높을수록 모델이 데이터 $x$를 조건 $c$에서 잘 이해하고 설명할 수 있다는 것을 의미합니다.

    $$
    P(x|c) = ∫P(x|z)P(z|c)dz
    $$

    하지만 Marginal Likelihood 직접 계산하는 것은 어려워 논문에서는 ELBO(Evidence Lower Bound)를 사용하여 간접적으로 $P(x|c)$를 최대화합니다.

    계산하기 어려운 이유 : $z$의 차원이 높거나 $P(x|z)$, $P(z|c)$가 복잡할 경우 적분하기가 어렵다.

2. ELBO(Evidence Lower Bound)

$$
\log p_\theta(x|c) \geq \mathbb{E}_{q\phi(z|x)} \Big[ \log p_\theta(x|z) - \log \frac{q_\phi(z|x)}{p_\theta(z|c)} \Big]
$$

-   $p_\theta(z|c)$: 조건 $c$가 주어진 잠재 변수 $z$의 사전 분포.
-   $p_\theta(x|z)$: 데이터 $x$의 가능도 함수(Likelihood Function).
-   $q_\phi(z|x)$: 근사 사후 분포(Approximate Posterior Distribution).

ELBO는 두 가지 주요 항목으로 구성됩니다.

Reconstruction Term $(- \log p_\theta(x|z))$

잠재 변수 $z$로부터 데이터를 얼마나 잘 재구성하는지 평가. 즉 $z$가 데이터 $x$를 잘 설명하도록 학습

KL Divergence $(log q_\phi(z|x) - \log p_\theta(z|c))$

$q_ϕ(z∣x)$ (근사 사후 분포)와$P_{\theta}(z∣c)$ (사전 분포) 간의 차이를 측정. $q_ϕ(z∣x)$가 $P_{\theta}(z∣c)$와 비슷해지도록 만들어 $z$가 조건 $c$를 잘 반영하도록 학습

---

### 2.1.2 Reconstruction Loss

Reconstruction Loss에서 타겟 데이터로 raw waveform 대신 **mel-spectrogram** $x_\text{mel}$ 을 사용합니다. 잠재 변수 $z$ 를 디코더를 통해 waveform 도메인 $\hat{y}$ 으로 업샘플링한 후, 이를 mel-spectrogram 도메인 $\hat{x}_\text{mel}$ 으로 변환합니다. 이후, 예측값과 타겟 mel-spectrogram 간의 $L_1$ 손실을 사용합니다

$$
\mathcal{L}_{recon} = \|x_{mel} - \hat{x}_\text{mel}\|_1
$$

이는 **최대 가능도 추정 (Maximum Likelihood Estimation)**으로 볼 수 있으며, mel 스케일은 인간 청각 시스템의 반응을 근사하므로 지각 품질을 향상시킵니다.

---

### 2.1.3 KL Divergence

KL Divergence은 다음과 같이 정의됩니다:

$$
\mathcal{L}_{kl} = \log _{q\phi}(z|x_\text{lin}) - \log _{p_\theta}(z|c_\text{text}, A)
$$

-   $q_\phi(z|x_\text{lin}) = N(z; \mu_\phi(x_\text{lin}), \sigma_\phi(x_\text{lin}))$: 잠재 변수 $z$ 의 정규 분포를 나타냄

### **2.1.4 정규화 흐름(Normalizing Flow,** $f_{\theta}$**)**

사전(prior) 및 사후(posterior) 인코더를 매개변수화하기 위해 분리된 정규 분포(factorized normal distribution)를 사용합니다. 정규화 흐름 $f_{\theta}$는 단순한 분포를 더 복잡한 분포로 변환하는 메커니즘입니다. 이를 통해 사전 분포의 표현력을 증가시키며 현실적인 샘플 생성이 가능해집니다.

$$
p_\theta(z|c) = N(f_\theta(z); \mu_\theta(c), \sigma_\theta(c)) \Big| \det \frac{\partial f_\theta(z)}{\partial z} \Big|


$$

$c=[c_{text},A]$ 는 텍스트와 정렬 정보를 포함합니다.

### **정렬 정보란?**

TTS에서는 입력된 텍스트가 음성 데이터와 시간적으로 어떻게 **정렬**(Alignment)되는지가 중요합니다

입력된 텍스트가 "안녕하세요"라고 가정해 보면, 이 텍스트를 음성으로 변환할 때

-   "안"은 몇 초 동안 지속되어야 하는가?
-   "녕"은 어느 위치에서 시작되고 얼마나 지속되는가?
-   "하"와 "세요"는 각각 얼마나 길게 발음되어야 하는가?

여기서 각 음소(예: "안", "녕", "하", "세", "요")가 음성 신호의 특정 시간 구간과 매핑되는 즉, 같은 **음소(phoneme)**와 **음성 데이터**의 매핑 관계를 **정렬 정보**라고 합니다.

2.2 Alignment Estimation

2.2.1 Monotonic Alignment Search

입력 텍스트와 타겟 음성 간의 정렬 $A$를 추정하기 위해, 논문에서는 Monotonic Alignment Search을 채택했습니다. MAS는 정규화 흐름 $f$로 매개변수화된 데이터의 가능도를 최대화하는 정렬을 탐색하는 동적 프로그래밍 방법입니다. MAS는 Monotonicity과 Non-skipping 두 가지 제약 조건을 만족하는 최적의 정렬 경로를 찾습니다.

**Monotonicity**

-   음소의 순서가 시간적으로 항상 증가해야 합니다.
-   예: "안" → "녕" → "하"

**Non-skipping**

-   모든 음소는 음성 프레임과 매핑되어야 하며, 프레임을 건너뛸 수 없습니다.

$$
A = \arg\max_{\hat{A}} \log p(x|c_\text{text}, \hat{A}) = \arg\max_{\hat{A}} \log N(f(x); \mu(c_\text{text}, \hat{A}), \sigma(c_\text{text}, \hat{A}))


$$

그러나 VITS는 Marginal Likelihood를 최대화하는 것이 아니라 ELBO를 최대화해서 Marginal Likelihood에 근사하는 것이 목적이기에, MAS를 그대로 사용하는 것이 아니라 다음과 같이 변형해서 사용합니다.

$$
\arg \max_{\hat{A}} \log P_\theta(x_\text{mel} \mid z) - \log \frac{q_\phi(z \mid x_\text{lin})}{p_\theta(z \mid c_\text{text}, \hat{A})}
$$

$$
 = \arg \max_{\hat{A}} \log p_\theta(z \mid c_\text{text}, \hat{A})
$$

$$
= \log N(f_\theta(z); \mu_\theta(c_\text{text}, \hat{A}), \sigma_\theta(c_\text{text}, \hat{A}))
$$

### **2.2.2 텍스트로부터 길이 예측(Duration Prediction from Text)**

각 입력 토큰 $d_i$의 길이는 추정된 정렬 행렬 $A$의 각 행에서 모든 열의 합으로 계산할 수 있습니다.

이걸 통해서 텍스트를 기반으로 발음의 길이를 어떻게 추정하고 학습하느냐를 다룰 수 있습니다.

$$
d_i = \sum_j A_{i,j}
$$

-   $A_{i,j}$: 텍스트의 $i$-번째 음소가 음성의 $j$-번째 프레임에 매핑될 가능도.
-   $d_i$: $i$-번째 음소의 총 길이는 정렬 행렬 $A$의 해당 행에서 모든 열의 값을 합한 값으로 계산됩니다.

### **Duration Calculation**

텍스트의 각 음소의 길이는 **정렬 행렬 $A$**를 사용해 계산됩니다.

### **예시**

-   텍스트: "안녕하세요"
-   정렬 행렬 $A$
    $A =
    \begin{bmatrix}
    0.9 & 0.8 & 0.1 & 0.0 & 0.0 \\
    0.1 & 0.2 & 0.8 & 0.7 & 0.0 \\
    0.0 & 0.0 & 0.1 & 0.2 & 0.8 \\
    \end{bmatrix}$

**Calculation**

-   "안" ($i=1$)
    $d_1 = 0.9 + 0.8 + 0.1 = 1.8 \, (\text{길이 1.8})$
-   "녕" ($i=2$)
    $d_2 = 0.1 + 0.2 + 0.8 + 0.7 = 1.8 \, (\text{길이 1.8})$
-   "하" ($i=3$):
    $d_3 = 0.0 + 0.0 + 0.1 + 0.2 + 0.8 = 1.1 \, (\text{길이 1.1})$

---

### **Deterministic vs Stochastic Duration Predictor**

1. **Deterministic Duration Predictor**
    - $d_i$는 항상 동일하게 예측됩니다.
    - 문제점: 사람이 말할 때 속도와 리듬이 달라질 수 있어 자연스러움 부족.
2. **Stochastic Duration Predictor**
    - 음소 길이를 **확률 분포**에서 샘플링하여 다양한 길이를 생성합니다.
    - 예: "안"의 길이가 평균 1.5초, 표준편차 0.2초인 분포에서 샘플링
        - 한 번은 1.6초, 또 다른 경우 1.4초로 다양하게 예측 가능

---

2.2.2 Stochastic Duration Predictor

VITS에서 Stochastic Duration Predictor는 텍스트의 음소 길이 $d$ 를 모델링하는 중요한 역학을 합니다. 이는 길이 값을 단순히 결정론적으로 예측하는 대신, 확률 분포에서 샘플링해서 더 자연스러운 발음을 생성합니다.

### 결정론적 접근 VS 확률적 접근

| **결정론적 접근 (CNN 학습 유사)**      | **확률적 접근 (Stochastic)**                |
| -------------------------------------- | ------------------------------------------- |
| 항상 고정된 출력 반환                  | 분포에서 샘플링하여 다양한 출력 생성        |
| 단순하고 직관적                        | 더 복잡하지만 자연스러움 표현 가능          |
| 데이터 다양성 반영 어려움              | 데이터 다양성과 불확실성 표현 가능          |
| 네트워크가 손실 함수를 최소화하며 학습 | 분포를 학습하며 샘플링 기반으로 다양성 제공 |

항상 동일한 길이를 반환하여 음성이 기계적으로 들리는 결정론적 접근의 한계로 활률적 길이 예측기를 사용

그러나, 확률적 길이 예측기에는 길이에 대한 몇 가지 문제점이 존재합니다.

1. 길이가 이산 정수값 (Discrete integer)

    음소 길이 $d$는 정수값, 그러나 정규화 흐름($f_\theta$)은 연속적인 실수 값에서 작동하므로 **비양자화(dequantization)** 가 필요

2. 길이가 스칼라(scalar) 값

    음소 길이 값은 단일 숫자(scalar)로 표현됩니다. **가역 변환(invertible transformation)**을 적용하려면 고차원 표현이 필요

위 문제를 해결하기 위해 VITS는 두 가지 기법을 사용합니다

1. Variational Dequantization

    랜덤 변수 $u$를 도입하여 $d - u$를 연속적인 실수 값으로 변환합니다.

    $u$는 $[0,1)$ 범위를 가지며, $d$의 불연속성을 제거합니다.

2. Variational Data Augmentation

    추가 랜덤 변수 $ν$를 도입하여 $d$와 결합합니다.

    $d$와 $ν$를 채널 단위로 결합해 고차원 잠재 표현을 만듭니다.

결과적으로, $u$와 $ν$를 통해 $d$**를 더 복잡한 분포로 표현**할 수 있게 됩니다.

### **결론**

-   **정렬 행렬** $A$ 를 기반으로 음소의 길이를 계산하고, 확률적 길이 예측기를 통해 더 자연스러운 발음 리듬을 생성합니다.
-   이는 사람이 매번 다른 리듬으로 발음하는 특성을 모델링하는 데 중요합니다.

### **2.3 Adversarial Training**

VITSA 모델은 **Adversarial Training**을 통해 음성 생성 품질을 향상시키고 자연스러운 음성을 학습합니다. 이를 위해 **Least-Squares Loss**와 **Feature-Matching Loss** 라는 두 가지 Loss를 사용합니다.

2.3.1 Discriminator Loss ($L_{adv}(D)$)

판별자 $D$의 역할은 **생성된 음성 $G(z)$**과 **실제음성 $y$** 를 구별

$$
L_{adv}(D)=\mathbb{E}_{(y,z)}[(D(y)−1)^2+D(G(z))^2]
$$

$Where:$

$D(y)$: 실제음성 $y$에 대한 판별기 출력

값이 $1 (True)$에 가깝도록 학습

$D(G(z))$: 생성된 음성 $G(z)$에 대한 판별기 출력

값이 $2(Fake)$에 가깝도록 학습

-   $D(y)$와 $D(G(z))$간의 차이를 Least-Squares Loss로 부드럽게 학습하여 학습 안정성을 높임
-   GAN Loss(Binary Cross Entropy)에 비해서 Model Collapsing을 방지하는 효과가 있음

    2.3.2 Generator Loss($L_{adv}(G)$)

생성기 $G$의 목표는 판별자 $D$를 속여 **생성된 음성 $G(z)$**를 실제 음성으로 구별하지 못하도록 하는 것

$$
L_{adv}(G)=\mathbb{E}_z[(D(G(z))−1)^2]
$$

$Where:$

$D(G(z))$: 생성된 음성에 대한 판별자의 출력

생성기가 $G(z)$를 진짜로 보이게 만들어야 해서 $D(G(z))$를 1에 가깝게 만듦

2.3.3. Feature-Matching Loss ($L_{fm}(G)$)

특징 매칭 손실은 생성음성과 실제음성 간의 중간 표현 차이를 줄이기 위한 손실

음성의 고수준과 저수준 특징 모두에서 생성음성이 실제음성과 유사하도록 학습

단순히 음성을 진짜로 보이게 하는 것이 아닌, 음성의 세부적인 구조와 특성을 유사하게 만듦

$$

L_{\text{fm}}(G) = \mathbb{E}_{(y, z)} \left[ \sum_{l=1}^T \frac{1}{N_l} \| D_l(y) - D_l(G(z)) \|_1 \right]


$$

$Where:$

$D_l(y)$: 실제음성 $y$에 대한 판별자의 $l$-번째 레이어에서 추출된 특징 맵

$D_l(G(z))$: 생성된 음성 $G(z)$에 대한 판별자 $l$-번째 레이어에서 추출된 특징 맵

$T$: 판별자의 총 레이어 수

$N_l$: $l$-번째 레이어의 특징 개수

2.4 Final Loss

VITS에서 **생성기 $G$**는 다음 손실을 최적화합니다

$$
L_{vae}=L_{recon}+L_{kl}+L_{dur}+L_{adv}(G)+L_{fm}(G)
$$

-   $L_\text{recon}$: 멜-스펙트로그램 기반의 음성 재구성 손실
-   $L_\text{kl}$: 잠재 변수 z의 분포를 정렬하는 KL 발산
-   $L_\text{dur}$: 확률적 길이 예측 손실
-   $L_\text{adv}(G)$: 생성기가 "진짜" 음성을 생성하도록 유도하는 적대적 손실
-   $L_\text{fm}(G)$: 생성 음성과 실제 음성 간의 특징 차이를 줄이는 손실

---

Architecture

### **1. 사전 인코더 (Prior Encoder)**

### **역할**

-   **텍스트 입력**과 **정렬 정보**를 기반으로 잠재 변수 $z$의 **사전 분포(prior distribution)**를 생성.
-   **추론 시** 텍스트 입력을 기반으로 $z$를 샘플링하여 음성을 생성.

### **구성 요소**

1. **텍스트 인코더**:
    - 텍스트 입력 ($c_{text}$)을 처리하여 **숨겨진 표현(hidden representation)** $h_{text}$ 생성
    - Transformer Encoder를 사용
2. **정규화 흐름 (Normalizing Flow)**:
    - $h_{text}$을 기반으로 사전 분포를 더 복잡하고 유연하게 모델링
    - **4개의 Affine Coupling Layer**로 구성
    - 각 Coupling Layer는 **4개의 WaveNet Residual Blocks**을 포함
    - **Volume-Preserving Transformation**으로 제한되어 스케일 파라미터를 생성하지 않음

---

### **2. 사후 인코더 (Posterior Encoder)**

### **역할**

-   실제 데이터 (선형 스케일 로그 스펙트로그램 $x_{lin}$)에서 잠재 변수 z의 **(approximate posterior distribution)**를 추출.
-   **훈련 시**에만 사용되어 재구성 손실 계산에 활용됨

### **구성 요소**

1. **입력 데이터**
    - 실제 음성을 선형 스케일 스펙트로그램 ($x_{lin}$)으로 변환하여 입력
2. **WaveNet 기반 구조**
    - **16개의 WaveNet Residual Blocks**으로 구성
    - 입력 데이터를 기반으로 잠재 변수 $z$를 생성
3. **Output**
    - 192 Channel 의 잠재 변수 $z$

---

### **3. 주요 차이점**

| 특성             | **사전 인코더 (Prior Encoder)**                                             | **사후 인코더 (Posterior Encoder)**                 |
| ---------------- | --------------------------------------------------------------------------- | --------------------------------------------------- |
| **사용 목적**    | 텍스트 기반 사전 분포 생성                                                  | 실제 데이터를 기반으로 잠재 변수 분포 생성          |
| **훈련 시 사용** | ELBO 계산의 KL 발산에 사용됨                                                | ELBO 계산의 재구성 손실과 KL 발산에 사용됨          |
| **추론 시 사용** | 텍스트 입력으로부터 $z$를 샘플링하여 음성을 생성                            | 사용되지 않음                                       |
| **구조**         | - Transformer Encoder- 4개의 Affine Coupling Layer- WaveNet Residual Blocks | 16개의 WaveNet Residual Blocks                      |
| **입력 데이터**  | 텍스트 입력($c_{text}$​)와 정렬 정보 ($A$)                                  | 선형 스케일 스펙트로그램 ($x_{lin}​$).              |
| **Output**       | 잠재 변수 $z$의 **사전 분포** $p_θ(z∣c_{text},A)$                           | 잠재 변수 $z$의 **근사 사후 분포** $q_ϕ(z∣x_{lin})$ |

---

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/e891e8e2-77a2-429a-a5aa-64d52c2b477e/dc4fb08e-3c18-4f71-b08a-c07e503fc039/image.png)

# 3. Experiments

## 3.1 Datasets

| **Dataset**   | **Details**                                                             |
| ------------- | ----------------------------------------------------------------------- |
| **LJ Speech** | - Single speaker dataset.                                               |
|               | - 13,100 short audio clips (~24 hours).                                 |
|               | - Audio format: 16-bit PCM, 22 kHz sample rate.                         |
|               | - Train: 12,500 samples, Validation: 100 samples, Test: 500 samples.    |
| **VCTK**      | - Multi-speaker dataset.                                                |
|               | - 109 native English speakers with various accents (~44 hours).         |
|               | - Audio format: 16-bit PCM, 44 kHz sample rate (downsampled to 22 kHz). |
|               | - Train: 43,470 samples, Validation: 100 samples, Test: 500 samples.    |

---

## 3.2 Preprocessing

| **Process**            | **Details**                                                                        |
| ---------------------- | ---------------------------------------------------------------------------------- |
| **Linear Spectrogram** | - Obtained from raw waveforms using Short-Time Fourier Transform (STFT).           |
|                        | - FFT size: 1024, Window size: 1024, Hop size: 256.                                |
| **Mel Spectrogram**    | - Reconstruction loss computed using 80-band mel-scale spectrogram.                |
|                        | - Mel-filterbank applied to linear spectrogram.                                    |
| **Phoneme Conversion** | - Text converted to International Phonetic Alphabet (IPA) using open-source tools. |
|                        | - IPA sequences interspersed with blank tokens (following Glow-TTS).               |

---

## 3.3 Training

| **Parameter**          | **Value/Details**                                                      |
| ---------------------- | ---------------------------------------------------------------------- |
| **Optimizer**          | AdamW with $\beta_1 = 0.8, \beta_2 = 0.99$, Weight decay: 0.01.        |
| **Learning Rate**      | Initial: $2 \times 10^{-4}$, Decay: $\times 0.999^{1/8}$ per epoch.    |
| **Batch Size**         | 64 per GPU (4 NVIDIA V100 GPUs).                                       |
| **Steps**              | Up to 800k steps with mixed precision training.                        |
| **Windowed Generator** | - Randomly extracted latent representation segments (window size: 32). |
|                        | - Corresponding raw waveform segments used as training targets.        |

---

## 3.4 Experimental Setup for Comparison

| **Model**                 | **Details**                                                                           |
| ------------------------- | ------------------------------------------------------------------------------------- |
| **Tacotron 2 + HiFi-GAN** | Two-stage system; Fine-tuned HiFi-GAN for Tacotron 2 and Glow-TTS for higher quality. |
| **Glow-TTS + HiFi-GAN**   | Non-autoregressive TTS system for comparison.                                         |
| **VITS**                  | Evaluated in single-speaker and multi-speaker settings.                               |

---

# 4. Results

## 4.1 Speech Synthesis Quality

| **Model**                              | **MOS (CI)**    |
| -------------------------------------- | --------------- |
| **Ground Truth**                       | $4.46 \pm 0.06$ |
| **Tacotron 2 + HiFi-GAN**              | $3.77 \pm 0.08$ |
| **Tacotron 2 + HiFi-GAN (Fine-tuned)** | $4.25 \pm 0.07$ |
| **Glow-TTS + HiFi-GAN**                | $4.14 \pm 0.07$ |
| **Glow-TTS + HiFi-GAN (Fine-tuned)**   | $4.32 \pm 0.07$ |
| **VITS (DDP)**                         | $4.39 \pm 0.06$ |
| **VITS**                               | $4.43 \pm 0.06$ |

---

## 4.2 Synthesis Speed

| **Model**               | **Speed (kHz)** | **Real-time Factor** |
| ----------------------- | --------------- | -------------------- |
| **Glow-TTS + HiFi-GAN** | 606.05          | $\times 27.48$       |
| **VITS**                | 1,480.15        | $\times 67.12$       |
| **VITS (DDP)**          | 2,005.03        | $\times 90.93$       |
