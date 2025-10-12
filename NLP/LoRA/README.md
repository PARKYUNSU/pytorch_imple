# LoRA (Low-Rank Adaptation) PyTorch 구현

LoRA(Low-Rank Adaptation of Large Language Models)는 대규모 언어 모델을 효율적으로 파인튜닝하기 위한 기법입니다. 이 프로젝트는 PyTorch를 사용하여 LoRA를 구현한 것입니다.

## LoRA 논문
LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS

[Read the Paper](https://arxiv.org/pdf/2106.09685)


LoRA는 LLM을 효율적으로 Fine-tuning하기 위한 방법으로, 전통적인 Fine-tuning과 다르게 기존 사전학습 모델의 가중치는 고정시키고, 각 층(Layer)에 저랭크 행렬(rank decomposition matricses)만 추가 학습합니다.

- 원래 가중치 $W_0$는 **고정(frozen)**

- 대신 $W_0 + \Delta W$ 형태로 학습

- $\Delta W = B A$ (저랭크 행렬 분해, rank $r$)

- $A \in \mathbb{R}^{r \times d}$, $B \in \mathbb{R}^{d \times r}$

이를통해서,

- **메모리 효율성**: 전체 모델을 파인튜닝하는 것보다 훨씬 적은 메모리 사용
- **학습 속도**: 업데이트해야 할 파라미터 수가 적어 빠른 학습
- **성능 유지**: 전체 파인튜닝과 유사한 성능 달성

# 1. Introduction
## 1.1. Background
NLP를 pre-training 에서 특정 태스크를 위한 fine-tuning이 일반적인 패러다임이 되었습니다.

그러나 GPT-3(175B) 같은 모델에서는 모든 파라미터를 업데이트하는 Full fine-tuning은 비현실적입니다.

## 1.2. Prior research & Limitations
효율성과 성능 간 트레이드 오프가 생겼던 선행 연구

1. Adapters
각 Transformer 블록에 작은 MLP(병목층)을 삽입하는 방법

- 파라미터가 적음 그러나 추론 속도가 느려짐 (inference latency 증가)
- 순차적으로 계산되어 병렬화가 어려움

2. Prompt Tuning(프롬프트 임베딩 학습)
큰 사전학습 모델은 동결, 테스크별로 작은 파라미터(프롬프트 임베딩)을 학습

- 원본 모델은 그대로 유지(동결)
- 학습 난이도가 높음(최적화 불안정)
- 프롬프트 길이가 늘어나면 실제 입력 토큰 길이가 줄어듦


## 1.3. 착안점
1. Li et al. (2018), Aghajanyan et al. (2020)
→ “모델이 학습되는 변화(gradient update)는 실제로 저차원(low intrinsic dimension) 공간에 존재한다.”

**가정:**
> "모델이 새로운 작업에 적응할 때, 가중치 변화($\Delta W$) 자체도 저랭크(low-rank) 구조를 가진다."

이 가정에 따라, 각 층의 가중치 업데이트를 저랭크 행렬 $B A$로 표현하여 학습합니다.

- $W_0$: 사전학습된 가중치 (**고정**)

- $\Delta W = B A$: 학습되는 **저랭크 업데이트** (rank $r$)

- 입력 $x$에 대한 출력:

$$
W_0 x + B A x
$$

- 초기화:

$$
A \sim \mathcal{N}(0, \sigma^2), \quad B = 0 \Rightarrow \text{학습 시작 시 } \Delta W = 0
$$

# 2. Problem Statement

사전 학습된 언어모델(ex. Transformer기반 GPT) $P_\Phi(y \mid x)$ (파라미터 $\Phi$)가 있다고 가정합니다.

다운스트림 작업(요약, MRC(기계독해), NL2SQL(자연어→SQL) 등)은 컨텍스트-타킷 쌍인 데이터셋 $Z = \lbrace (x_i, y_i) \rbrace_{i=1}^N$로 주어집니다.

## 2.1. Full fine-tuning

모델을 사전학습 가중치 $\Phi_0$로 초기화하고, 아래 조건부 언어모델링 목적함수를 최대화하도록 $\Phi$를 업데이트합니다.

**표기법:**
- $P_\Phi(y \mid x)$: 사전학습 언어모델(autoregressive)
- $Z = \lbrace (x_i, y_i) \rbrace$: 컨텍스트-타깃 쌍 데이터셋
- $y_{< t}$: 시점 $t$ 이전의 타깃 토큰들
- $\Phi_0$: 사전학습 가중치
- $\Delta\Phi$: 작업별 가중치 증가분
- $\Theta$: $\Delta\Phi$를 부호화하는 소수 파라미터 집합 ($\vert\Theta\vert \ll \vert\Phi_0\vert$)

**목적함수:**

$$
\max_\Phi \sum_{(x,y) \in Z} \sum_{t=1}^{\vert y \vert} \log P_\Phi(y_t \mid x, y_{< t}) \quad \cdots (1)
$$

**단점:** 작업마다 $\Delta\Phi$를 따로 학습, 저장해야 하고, 그 크기 $\vert\Delta\Phi\vert$가 $\vert\Phi_0\vert$와 동일함. 대형 모델일수록(ex. GPT-3 175B) 여러 작업 인스턴스를 배포하기 어렵거나 비현실적.

## 2.2. PEFT(Parameter-Efficient Fine-Tuning)

작업별 증가분 $\Delta\Phi$를 **작은 파라미터 집합 $\Theta$**로 부호화합니다. 즉, $\Delta\Phi = \Delta\Phi(\Theta)$이며 $\vert\Theta\vert \ll \vert\Phi_0\vert$입니다. 이제 최적화 대상은 $\Phi$가 아니라 $\Theta$입니다.

$$
\max_\Theta \sum_{(x,y) \in Z} \sum_{t=1}^{\vert y \vert} \log P_{\Phi_0 + \Delta\Phi(\Theta)}(y_t \mid x, y_{< t}) \quad \cdots (2)
$$

이때 $\Delta\Phi$는 저랭크(low-rank) 표현으로 구성해 연산/메모리 효율을 얻습니다. 예를 들어 GPT-3 175B 기준으로 학습 파라미터 수 $\vert\Theta\vert$를 $\vert\Phi_0\vert$의 약 0.01%까지 줄일 수 있습니다.


# 3. Aren't Existing Solutions Good Enough? (기존 방법들이 충분하지 않은 이유)

## 3.1. Existing Solutions

| 구분 | 대표 기법 | 핵심 아이디어 |
| --- | --- | --- |
| ① Adapter Layers | Houlsby et al., 2019 | 각 Transformer 블록에 작은 MLP 층을 삽입하여 태스크 별 파라미터만 학습 |
| ② Prompt Optimization | Prefix-Tuning, Prompt-Tuning | 입력 프롬프트에 학습 가능한 "가상 토큰"을 추가하여 모델 출력 유도 |

## 3.2. Adapter Layers 문제점

Adapter는 파라미터 수가 작지만, 추론 속도를 느리게 만듭니다. 각 Transformer block 마다 adapter층을 추가하면, 계산 순서가 깊어져서 병렬화가 안 됨, 특히 실시간 inference(Batch size=1)에서는 속도가 지연됨.

> 작은 모델이라도 Adapter를 쓰면 실시간 응답 속도가 최대 30% 느려짐. 또한 모델을 여러 GPU에 샤딩하면, 통신 오버헤드가 추가됨.

## 3.3. Prompt Optimization 문제점
입력 임베딩을 직접 학습하 Pormpt Tuning은 다음과 같은 문제점이 있습니다.

- 최적화가 불안정함 → 성능이 일관되지 않음
- 학습 토큰 수를 늘리면 성능이 오히려 하락
- 모델의 실제 입력 시퀀스 길이가 줄어듦
(“적응용 토큰”이 길이를 차지하기 때문)

결과적으로 prompt 기반 방법은 “입력 길이 감소 → 성능 저하” 문제가 발생합니다.

## 3.4. LoRA
선행 연구되었던 두 방법은 느리고, 불안정 하기에 효율적인 튜닝 방법이 필요했으며, 연구진들은 LoRA 기법으로 두 간극을 메우는 방법을 제시합니다.

## 3.5. 정리
| 기법               | 추가 파라미터 | 학습 안정성 | 추론 속도 | 성능       |
| ---------------- | ------- | ------ | ----- | -------- |
| Full Fine-Tuning | 100 %   | 안정적    | 빠름    | 최고       |
| Adapters         | < 1 %   | 안정     | 느림    | 좋음       |
| Prefix Tuning    | < 1 %   | 불안정    | 빠름    | 불안정      |
| **LoRA (제안)**    | < 0.1 % | 안정적    | 빠름    | 동등 또는 상회 |


# 4. Our Method(LoRA)

## 4.1. Intuition

논문의 저자들은 **"모델의 파라미터 변화량($\Delta W$)은 저랭크 구조를 가진다"**라는 가정을 합니다.

즉, 대규모 언어모델의 전체 파라미터 수는 수십억 단위지만, 실제로 다운스트림 태스크에 적응할 때 필요한 정보는 훨씬 적다고 가정합니다. 전체 파라미터 공간 중 일부 저차원 공간만 수정되어도 충분하다는 뜻입니다.

이를 수학적으로 표현하면 다음과 같습니다:

$$
W = W_0 + \Delta W, \quad \text{where } \Delta W = B A
$$

**표기:**
- $W_0$: 사전학습(pre-trained)된 가중치 (고정, freeze)
- $A \in \mathbb{R}^{r \times d}$, $B \in \mathbb{R}^{d \times r}$
- $r \ll d$: 매우 작은 rank (예: $d=4096$, $r=4$ 또는 $8$ 정도)

즉, LoRA는 $\Delta W$를 두 개의 저차원 행렬로 분해해서 학습합니다.  
→ 이 구조가 바로 **"Low-Rank Adaptation (저랭크 적응)"**의 본질입니다.

## 4.2. 수식으로 본 구조

Transformer에서 Linear 연산은 다음과 같습니다:

$$
h = W x
$$

여기서 LoRA를 적용하면:

$$
h = W_0 x + \Delta W x = W_0 x + B A x
$$

**역할:**
- $A$: **down projection** (차원 축소)
- $B$: **up projection** (차원 확장)

즉, $A \to B$는 **"저차원 병목"** 경로를 형성합니다.  
(Autoencoder처럼 중간에 저랭크 병목을 만들어 특이성만 학습하는 구조)

## 4.3. 초기화 방식

학습 안정성을 위해 LoRA는 다음처럼 초기화합니다:

$$
A \sim \mathcal{N}(0, 0.01), \quad B = 0
$$

이때 학습 초기에는 $\Delta W = B A = 0$이므로, 초기 출력은 원래 모델의 출력 $W_0 x$과 완전히 동일합니다.

즉, LoRA는 학습 시작 시점부터 원본 모델의 성능을 그대로 유지하며, 훈련이 진행될수록 점진적으로 태스크에 맞게 적응하게 됩니다.

** 장점:**
- Catastrophic Forgetting 없음
- Fine-tuning 초반 불안정 현상 방지
- "안정적인 시작점" 확보

## 4.4. Scaling (확대 계수 $\alpha$)

LoRA는 rank가 매우 작기 때문에, $B A$가 만들어내는 $\Delta W$의 스케일(값 크기)이 너무 작을 수 있습니다.

이를 보정하기 위해 scaling factor $\alpha$를 곱합니다:

$$
\Delta W = \frac{\alpha}{r} B A
$$

- $\alpha$는 보통 rank $r$보다 조금 큰 상수 (예: $\alpha=8$, $r=4$)
- 따라서 $\frac{\alpha}{r}$는 2배 정도의 스케일 조정

이 조정 덕분에 rank가 작아도 학습 효과가 충분히 반영됩니다.

**💡 LoRA의 핵심 하이퍼파라미터:**
- **rank $r$**: 얼마나 낮은 차원으로 병목을 만들지
- **scaling $\alpha$**: 업데이트 강도 (일종의 학습율 조정 효과)

## 4.5. 어디에 적용되는가 (Which Layers to Apply?)

LoRA는 Transformer 내에서 모든 Linear Layer에 적용할 수 있지만, Self-Attention의 **Query ($W_q$)**와 **Value ($W_v$)** 가중치에만 적용하는 것이 가장 효율적임을 보였습니다.

**이유:**
- $W_q$, $W_v$는 attention head마다 독립적으로 존재
- **쿼리(Query)**: 어떤 정보에 집중할지를 결정
- **값(Value)**: 그 정보를 어떻게 활용할지를 결정
- 즉, 모델이 "새로운 작업에 적응"할 때 주로 변하는 부분은 이 두 행렬

**반면:**
- **Key(키)**: 주로 정적 역할
- **Output($W_o$), Feed-forward(FFN)**: 변화의 영향이 적음

**실험 결과:**
- $W_q$, $W_v$에만 LoRA를 적용해도 모든 가중치에 LoRA를 적용한 결과와 거의 동일한 성능
- 연산량과 파라미터 수가 더욱 줄어듦

## 4.6. LoRA의 계산량 분석

LoRA의 추가 연산량은 극히 미미합니다.

**기존 연산:**

$$
y = W_0 x
$$

**LoRA 추가:**

$$
y = W_0 x + B A x
$$

**계산 복잡도:**
- $A x$: $(r \times d) \times (d \times 1) = O(r \cdot d)$
- $B (A x)$: $(d \times r) \times (r \times 1) = O(d \cdot r)$

총 추가 계산량은 $O(r \cdot d)$,  
→ 전체 $O(d^2)$ 대비 무시할 정도로 작음 ($r \ll d$)

**예:** GPT-3의 $d=12288$, $r=8$일 때, 추가 계산량은 **0.06% 수준**.

## 4.7. 추론(Inference) 시 병합 (Merge)

LoRA의 또 하나의 강점은 **추론 시 속도 저하가 전혀 없다는 것**입니다.

학습이 끝난 뒤,

$$
W = W_0 + \frac{\alpha}{r} B A
$$

를 미리 계산해 병합(merge)합니다.

이렇게 하면 추론 시에는 LoRA가 존재하지 않는 것처럼, 원래 모델처럼 단일 $W$로 연산됩니다.

**결과:**
- 학습 시에는 저랭크 업데이트만 수행
- 추론 시에는 완전히 통합된 모델로 동작
- 속도, 메모리, 정확도 모두 손실 없음





---
## LoRA의 행렬 분해 형태

- 선형층 가중치 $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$는 **동결**.
- 학습하는 변화량은 저랭크 행렬 곱으로 표현:

$$
\Delta W = B A,\qquad 
B \in \mathbb{R}^{d_{\text{out}} \times r},\ 
A \in \mathbb{R}^{r \times d_{\text{in}}},\ 
r \ll \min(d_{\text{out}}, d_{\text{in}})
$$

- 최종 가중치:

$$
W = W_0 + \Delta W = W_0 + B A
$$

- 출력(입력 $x \in \mathbb{R}^{d_{\text{in}}}$에 대해):

$$
y = W x = W_0 x + (B A) x = W_0 x + B (A x)
$$

  여기서 $A$가 **down-projection** (차원 $d_{\text{in}} \to r$), $B$가 **up-projection** (차원 $r \to d_{\text{out}}$) 역할.

- **랭크 성질**: $\text{rank}(\Delta W) \le r$  
  (저랭크 업데이트이므로, 작은 $r$로도 충분히 표현력을 확보하는 것이 핵심 가정)

- **파라미터 수**: 기존 층은 $d_{\text{out}} \times d_{\text{in}}$이지만, LoRA는 $B$와 $A$만 학습  
  → 파라미터 수: $d_{\text{out}} \cdot r + r \cdot d_{\text{in}} = r(d_{\text{out}} + d_{\text{in}})$

- **초기화/스케일링(자주 쓰는 형태)**:

$$
A \sim \mathcal{N}(0,\sigma^2),\quad B=0 \Rightarrow \Delta W(\text{초기})=0
$$

$$
\Delta W = \frac{\alpha}{r}\, B A \quad (\text{스케일 조정})
$$

## SVD와의 관계(직관)

- 임의의 행렬 갱신 $\Delta W$에 대해 **최적의 랭크-$r$ 근사**는 SVD로 얻을 수 있음:

$$
\Delta W \approx U_r \Sigma_r V_r^{\top}
$$

- 이때 LoRA 형태로 **재매개변수화** 가능:

$$
B = U_r \Sigma_r,\quad A = V_r^{\top}
\quad\Rightarrow\quad
B A = U_r \Sigma_r V_r^{\top}
$$

  (실제 학습에서는 SVD를 매 스텝 구하지 않고, $A,B$를 직접 학습)

## 작은 예시(모양만)

- 예: $d_{\text{out}}=4,\ d_{\text{in}}=3,\ r=2$

$$
B=
\begin{bmatrix}
\bullet & \bullet \\
\bullet & \bullet \\
\bullet & \bullet \\
\bullet & \bullet
\end{bmatrix}_{4\times 2},\quad
A=
\begin{bmatrix}
\bullet & \bullet & \bullet \\
\bullet & \bullet & \bullet
\end{bmatrix}_{2\times 3}
$$

$$
\Delta W = B A \in \mathbb{R}^{4\times 3},\quad
W = W_0 + \Delta W
$$

## 합치기(추론용 병합)

- 학습 종료 후 하나로 병합:

$$
W_{\text{merged}} = W_0 + \frac{\alpha}{r} B A
$$

  ⇒ 추론 시엔 **일반 선형층과 동일 경로**로 계산(추가 지연 없음).




---
## 설치 방법

```bash
# 저장소 클론
git clone <repository-url>
cd LoRA

# 필요한 패키지 설치
pip install -r requirements.txt
```

## 사용 방법

```python
# 예제 코드
import torch
from lora import LoRALayer

# LoRA 레이어 생성
lora_layer = LoRALayer(
    in_features=768,
    out_features=768,
    rank=8
)

# 모델에 적용
# ... (추가 예정)
```

## 프로젝트 구조

```
LoRA/
├── README.md
├── requirements.txt
├── lora.py           # LoRA 레이어 구현
├── train.py          # 학습 스크립트
├── inference.py      # 추론 스크립트
└── examples/         # 예제 코드
```

## 주요 기능

- [ ] LoRA 레이어 구현
- [ ] 다양한 모델 아키텍처 지원
- [ ] 학습 및 추론 파이프라인
- [ ] 사전 학습된 모델과의 통합

## 참고 자료

- [LoRA 논문](https://arxiv.org/abs/2106.09685): "LoRA: Low-Rank Adaptation of Large Language Models"
- [Hugging Face PEFT 라이브러리](https://github.com/huggingface/peft)

## 라이센스

MIT License

## 기여

이슈 제기 및 풀 리퀘스트를 환영합니다!


