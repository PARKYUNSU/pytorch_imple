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


