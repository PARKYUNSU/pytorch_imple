# SELF-RAG

"Learning to Retrive, Generate, and Critique through Self-Reflection"

ㅡ Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, Hannaneh Hajishirzi

[Paper to Read](https://arxiv.org/pdf/2310.11511)

# 1. Introduction

LLM은 기존에 내포하고 있는 Parameter 기반 지식에만 의존하다 보니 종종 사실과 다른 부정확한 응답을 생성하는 Hallucination 형상이 종종 일어났습니다. 그래서 Query에 관련된 지식 검색 방법으로 모델을 보강하는 Retrieval-Augmented Generation([RAG](https://github.com/PARKYUNSU/pytorch_imple/tree/main/Agentic_RAG/Basic_Agentic_RAG))는 이런 문제를 줄여주지만, 검색이 반드시 필요한지 여부나 검색된 구절들이 다 관련성이 있는지에 상관없이 무분별하게 검색하여 내용에 포함시키려고 해서 모델의 다재다능성을 저해하거나 결과적으로 도움이 되지 않는 답변을 생성할 수 도 있습니다.

이에 본 논문은 Retrieval과 Self-reflection을 통해서 모델의 품질과 사실성을 향상시키는 SELF-RAG(Self-Reflective Retrieval-Augmented Generation)을 소개합니다. 본 모델은 필요에 따라 구절을 적응적으로 검색하고, 검색된 구절 및 자체 생성된 텍스트에 대해 Special Tokens(통칭, reflection tokens)을 사용하여 생성 및 Reflection(반성)하는 임의의 단일 모델을 End-to-end 방식으로 학습시킵니다.

이 Special Tokens을 생성함으로써 추론 단계에서 모델의 동작을 제어할 수 있어서, 다양한 작업 요구사항에 맞추어 모델의 행동을 조정할 수 있어서 SOTA LLM 모델과 Retrieval-augmented 모델 보다 더 월등한 성능을 보입니다.

# 2. RAG VS SELF RAG
## 2.1. RAG
기존 [RAG](https://github.com/PARKYUNSU/pytorch_imple/tree/main/Agentic_RAG/Self_RAG)의 방식은 다음과 같습니다.

Query(Prompt)에 관련된 K개의 구절들을 문서들에서 찾아서 질문한 Query와 합쳐서 문장을 제구성 및 생성해서 사용자에게 제공하는 구조를 가집니다.

그러나 Contradictory(모순적)이고 No information(정보가 없는) 답변을 생성하는 RAG의 문제점을 아래 그림에서 확인 할 수 있습니다.

<img src="https://github.com/user-attachments/assets/527b3b7f-e3ea-4dfb-a941-ce81a23a49e4" width=400>

## 2.2. SELF-RAG
SELF-RAG는 필요에 따른 검색과 Self-Refelction을 통한 방법인 Special Tokens을 도입하면서 End-to-end 학습하여 모델의 생성 품질 및 정확도를 향상키는 모델이라 소개합니다.

Special Tokens(Reflection Tokens)은 검색의 필요성과 생성 품질을 각각 나타내기 위해 Retrieval Token과 Critique Token(Relevant, Irrelevant, Fully supported, Partially Supported, No support)로 분류됩니다.

SELF-RAG는 기존 RAG의 방식과 다른 방법으로 검색합니다.

- Step 1: Query (Prompt)을 모델을 통해 생성한 문장을 생성합니다. 그 이후 외부 문서나 정보를 참고하면 더 정확하거나 풍부한 답변을 만들 수 있다고 판단하면, Retrieval Token을 출력하여 검색기가 문서를 가져올 수 있게 신호를 줍니다.
- Step 2: SELF-RAG는 외부 정보에서 관련성을 평가하면서 구절을 선택하고 Output을 생성합니다.
- Step 3: SELF-RAG는 마지막으로 Output들에 대해 Critique Tokens을 생성하여 출력한 결과의 사실성 및 전반적인 품질을 평가하고 최적의 Output을 선택합니다.
  
<img src="https://github.com/user-attachments/assets/e6eb1a5f-54ab-4401-a2a4-cc7cc08903ea" width=500>

## 2.3. RAG vs SELF-RAG (Query No need to retrieve)
또한 외부 정보에서 검색의 필요성이 없는 경우에도 RAG는 아래 그림처럼 K개의 Documents를 검색해서 답변을 생성하려고 하는 반면에, SELF-RAG는 검색 없이 모델이 Query의 답변을 생성합니다.

![image](https://github.com/user-attachments/assets/5b58787a-12bf-4f64-a924-d49f43bbb659)


# 3. SELF-RAG
위에서 간단하게 기존 RAG와 다른 SELF-RAG의 검색 방법에 대해서 알아 봤습니다. LLM의 본래 지닌 역량은 해치지 않으면서 검색의 효율을 높이는 SELF-RAG의 End-to-end의 학습 방법을 알아보고자 합니다.

SELF-RAG는 Special Tokens을 생성하는 법을 학습함으로써 Output에 대해 Criticize를 할 수 있도록 LM(Language Model) $M$을 학습 시킵니다.

학습하게 되는 Special Tokens들은 아래 테이블에 정리되어있습니다.

## 3.1. Problem Formalization and Overview
SELF-RAG는 Input $x$가 주어지먄, LM $M$이 여러 Segment로 구성된 출력 $y = [y_1, ..., y_T]$를 순차적으로 생성하도록 학습합니다. $y_t$는 $t$번째 Segment의 Token 시퀀스로 기존 LLM이 사전 학습을 통해 익히 기본 Token과 Special Token이 포함되어있습니다.

<img src="https://github.com/user-attachments/assets/a704afb5-af41-4f80-a019-2a2a467aaa4c" width=800>

| 4개의 Reflection Tokens - 아래 3개는 Critique Tokens

<img src="https://github.com/user-attachments/assets/b7fc365e-003b-4764-b2c2-363e143ab071" width=700>

### 3.1.1. Inference Overview
SELF-RAG는 입력 $x$와 모델이 생성한 문장 $y_{<t}$를 바탕으로, 현재 이어서 생성할 텍스트에 외부 정보가 필요할지 판단합니다.
```text
Input x: "미국 주 이름은 어떻게 지어졌나요"

preceding generation : "미국 주 이름은 다양한 역사적 배경과 문화적 요소에서 기인합니다. 일부 주는 유명한 인물의 이름을 따르기도 하고, 일부는 지리적 특성을 반영합니다."
```

여기서 검색이 불필요하다고 하면 기존 LM처럼 다음 문장을 바로 생성합니다.

검색 필요 시, 모델은 검색된 구절의 관련성을 평가하기 위한 Critique Tokens[relevant, irrelevant]을 만들고 문장을 생성합니다.

생성된 문장에서도 검색된 구절이 충분히 지원하고 있는지 평가를 통해 다른 Critique Tokens[full supported, partially supported, no support] 생성합니다.

최종적으로, 전체 응답의 유용성을 평가하는 최종 Critique Tokens[5, 4, 3, 2, 1]를 생성합니다.

SELF-RAG는 응답들을 최종적으로는 점수화하여 필터링하여 제어하여 soft constraint와 Hard control를 적용할 수 있습니다.

### 3.1.2. Training Overview
SELF-RAG는 기존 LM의 기본 어휘 Tokens에 Special Tokens를 추가하여 학습을 진행합니다.

- Generator model $M$
Retriever $R$이 검색한 구절과 Critic model $C$가 예측한 Special Tokens이 교차로 삽입된 정제된 코퍼스를 사용해 학습됩니다.

- Critic model $C$
입력, 출력, 검색된 문서를 바탕으로 Special Tokens을 생성하도록 지도학습되며, 이를 통해 훈련 데이터에 Special Tokens이 오프라인으로 삽입됩니다.

최종적으로, 다음 Tokens을 생성하는 Generator model $M$을 학습하여, 추론 시에 별도의 Critic model 없이 $M$이 스스로 Speicial Tokens를 생성하고 활용하게 됩니다.

<img src="https://github.com/user-attachments/assets/a3395073-f137-45e6-bfe4-170b24411704" width=700>

## 3.2. SELF-RAG Training
### 3.2.1. Training The Critic Model
#### Critic 데이터 수집
Critic model을 훈련하기 위해 데이터를 수집해야지만, 수동 주석은 비용이 많이 들기 때문에, 우리는 GPT-4와 같은 최신 LLM을 사용하여 Special Tokens(reflection tokens)에 대한 피드백을 자동으로 생성합니다. 

그러나 이런 방법은 API 비용이 증가하는 단점이 있어서, GPT-4를 프롬프트하여 Special Tokens을 생성하도록 한 후, 그 지식을 내부 Critic Model $C$에 증류시키는 방식으로 지도학습 데이터를 생성합니다.

각 Special Tokens 그룹(예: Retrieve, ISREL, ISSUP, ISUSE)에 대해, 원래 훈련 데이터 ${X, Y}$에서 무작위로 인스턴스 ${X_{\text{sample}}, Y_{\text{sample}}}$를 샘플링합니다.

예시와 함께 “웹에서 외부 문서를 찾는 것이 도움이 되는지” 등의 지시문을 제공해 $p(r \mid I, x, y)$를 예측하고, 이를 $D_{\text{critic}}$에 저장합니다.

#### Critic 학습
사전 학습된 LM(예: Llama 2-7B)으로 $C$를 초기화한 후, $D_{\text{critic}}$ 데이터를 이용해 다음 토큰 예측 손실을 최소화합니다

$$ \max_{C}\mathbb{E}_{((x, y), r)\sim\text{D}{\text{critic}}}log{pC}(r|x,y)$$

이 방법으로 $C$는 GPT-4 피드백과 90% 이상의 일치율을 달성합니다.

### 3.2.2. Training The Generator Model
#### Generator 데이터 수집
입력-출력 쌍 $(x, y)$를 바탕으로, Retriever model $R$과 Critic model $C$를 활용해 원래 출력 $y$를 보강하여 SELF-RAG 추론 과정을 모방하는 지도 학습 데이터를 생성합니다.

출력 $y$의 각 세그먼트 $y_t$마다 $C$를 실행하여, 추가 구절이 생성 향상에 도움이 되는지 평가합니다.

만약 검색이 필요한 경우, Retrieve Special token인  **Retrieve = Yes**를 추가하고, $R$은 상위 $K$개의 구절 집합 $D$를 검색합니다.

각 구절에 대해, $C$는 해당 구절의 관련성(ISREL)을 평가하며, 관련성이 인정되면 그 구절이 생성 결과를 지지하는지(ISSUP)를 추가로 평가합니다.

마지막 세그먼트 $y_T$에서는 $C$가 전체 유용성(ISUSE)을 평가하여, Special Tokens이 포함된 보강된 출력과 원래 입력 쌍이 생성 데이터 집합 $D_{\text{gen}}$에 추가됩니다.

#### Generator 학습
- Special Tokens이 삽입된 보강된 코퍼스 $D_{\text{gen}}$을 사용하여,  Generator model $M$을 표준 다음 토큰 예측 목표로 학습합니다.

$$ \max_{M}\mathbb{E}_{(x, y, r)\sim\text{D}{\text{gen}}}log{pM}(y,r|x)$$

Critic model $C$를 학습하는 것과 달리, $M$은 목표 출력과 함께 Special Tokens도 예측하는 법을 학습합니다.

학습 과정에서는, `<p>`와 `</p>`로 둘러싸인 검색된 텍스트 조각들을 마스킹하고, 원래 어휘 $V$에 Special Tokens 집합 {Critique, Retrieve}을 추가하여 확장합니다.

#### Connections to prior work on learning with critique
최근 연구에서는 PPO를 통한 RLHF와 같이 훈련 중 추가 비판(피드백)을 도입합니다. PPO는 별도의 보상 모델에 의존하지만, SELF-RAG는 Critique를 오프라인으로 계산하여 훈련 코퍼스에 직접 삽입함으로써 훈련 비용을 크게 줄입니다.

또한, SELF-RAG는 생성된 각 세그먼트 후 자체 평가를 위한 Special Tokens을 생성하여, 추론 시 소프트 리랭킹 또는 하드 제약을 적용하는 방식의 생성 제어를 가능하게 합니다.


## 3.3. SELF-RAG Inference
#### 제어 가능 추론 
SELF-RAG는 생성된 출력에 대해 Special Tokens을 사용해 자체 평가를 수행하여, 추론 단계에서 모델의 동작을 제어합니다.  
  - Tasks Demanding Factual Accuracy: 출력이 증거와 밀접하게 일치하도록 검색을 더 자주 실행합니다.  
  - More Open-ended Tasks: 검색 빈도를 줄이고 전반적인 유창성과 유용성을 우선시합니다.

#### Adaptive retrieval with threshold
SELF-RAG는 Retrieve 토큰의 생성 확률을 정규화하여, 그 값이 지정된 임계값을 초과하면 검색을 실행합니다.

#### Tree-decoding with critique tokens
각 세그먼트 단계 $t$에서, 검색이 필요한 경우 Retriever $R$는 $K$개의 구절을 검색하고, Generator $M$은 이를 병렬로 처리하여 $K$개의 후보를 생성합니다.  
세그먼트 수준의 Beem Search를 통해 매 시점 $t$마다 상위 $B$개의 후보를 선택하고, 최종 시퀀스를 반환합니다. 각 세그먼트 $y_t$의 점수는 다음과 같이 계산됩니다.

$$
f(y_t, d, \text{Critique}) = p(y_t | x, d, y_{\text{<}t}) + S(\text{Critique})
$$

  
$$
S(\text{Critique}) = \sum_{G \in \{\text{ISREL}, \text{ISSUP}, \text{ISUSE}\}} w_G \, s_G^t
$$


$s_G^t$는 해당 그룹 $G$에 대해 가장 바람직한 Special Tokens $\hat{r}$의 생성 확률을 나타내며, $w_G$는 사용자가 조정할 수 있는 하이퍼파라미터입니다. 이 평가를 바탕으로, 바람직하지 않은 후보(예: ISSUP = No support)를 필터링할 수 있습니다.

최종적으로, SELF-RAG는 추가 훈련 없이도 추론 시 사용자가 원하는 방식으로 모델의 동작을 조정할 수 있습니다.


# 4. Experiments
## 4.1. Tasks and Datasets

SELF-RAG의 성능 평가를 위해 여러 유형의 작업과 데이터셋을 사용했습니다. 먼저, **Closed-set 작업**에서는 PubHealth와 ARC-Challenge와 같이 정답 기반 평가를 수행하여 모델의 정확도를 측정했습니다. 다음으로, **Short-form 생성 작업**에서는 PopQA와 TriviaQA-unfiltered를 통해 모델이 정답을 포함하는지를 확인하였고, 마지막으로 **Long-form 생성 작업**에서는 전기(biography) 생성과 장문 QA를 활용하여 FactScore, MAUVE, 인용 정밀도 및 재현율과 같은 다양한 평가 지표로 성능을 분석했습니다. 각 데이터셋의 특징과 평가 기준에 대해 간략하게 설명함으로써, SELF-RAG의 다양한 작업에 대한 적용 가능성을 제시합니다.

## 4.2. Baselines

SELF-RAG의 성능을 비교하기 위해 두 가지 범주의 Baseline 모델을 선정했습니다. 하나는 **검색 없이 사용하는 모델**으로, Llama, Alpaca, ChatGPT 등이 있으며, 다른 하나는 **검색 보강 모델**으로, 표준 RAG 기반 모델, Llama2-FT, Ret-ChatGPT 등이 포함됩니다. 각 모델의 주요 특징과 적용 방식, 그리고 SELF-RAG와의 차별점을 중심으로 비교 분석합니다.

## 4.3. Experimental Settings

실험 설정에서는 사용한 훈련 데이터, 모델 구성 및 하이퍼파라미터, 그리고 추론 방법에 대해 상세히 기술합니다. 훈련 데이터는 다양한 지침-출력 쌍으로 구성되었으며, 데이터셋 출처와 샘플 수를 명시합니다. 모델 구성에서는 Llama2-7B/13B와 같은 기반 모델, Beem Search 및 임계값 설정 등의 세부 사항을 포함하며, 추론 시에는 검색 빈도와 토큰 디코딩 방법 등이 적용되었습니다.

## 4.4. Results and Analysis

SELF-RAG의 성능과 다른 Baseline 모델과의 비교 결과를 종합적으로 제시합니다. 전체 작업에 대한 성능 비교(예: Table 2)를 통해 SELF-RAG가 왜 검색 보강 ChatGPT, Llama2-chat, Alpaca 등을 능가하는지 설명합니다. 또한, ablation studies를 통해 검색 필요성, 비판 토큰 등의 개별 구성 요소가 전체 성능에 미치는 영향을 분석하고, 추론 시 사용자 맞춤형 제어 효과(예: 인용 정밀도 및 MAUVE 점수 변화)를 상세히 검토합니다. 효율성과 정확도의 trade-off, 학습 데이터 규모의 영향, 그리고 인간 평가 결과를 포함한 분석도 함께 제시됩니다.


# 5. Result

### 전체 실험 결과
<img src="https://github.com/user-attachments/assets/54bfdd19-bb70-47f5-a558-81f8d443d3d5" width=800>

| Bold : The best perform among Non-proprietary models, Gray-Bold : The best proprietary model / FS :FactScore, em :correctness, mau: MAUVE(fluency)

### SELF-RAG 분석
<img src="https://github.com/user-attachments/assets/4a29b599-ca39-45e8-a4ad-07d382d53a47" width=600>

|(a) 7B 모델을 기반으로 SELF-RAG 훈련 및 추론의 핵심 구성 요소에 대한 ablation 연구 결과, (b) ASQA 인용 정밀도와 MAUVE(유창성)에 미치는 소프트 가중치의 효과, (c) PubHealth와 PopQA에서의 검색 빈도와 정규화된 정확도의 관계

### Training Scale and Human Analysis
<img src="https://github.com/user-attachments/assets/c323b90f-f30c-4003-a57c-730b118f0d09" width=600>

(a), (b), (c) Training Scale 분석은 PopQA, PubHealth, ASQA(인용 정밀도)에 대해 훈련 데이터 규모가 미치는 효과를 각각 보여줍니다.

(d) SELF-RAG의 출력 및 Special Tokens에 대한 인간 평가 결과를 제시합니다.
