# CRAG (Corrective RAG)

"Corrective Retrieval Augmented Generation" - 2024

ㅡ Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, Zhen-Hua Ling

[Read the Paper](https://arxiv.org/pdf/2401.15884)

---
# 1. Introduction

LLM의 대표적인 문제인 Hallucination(환각) 현상은 선행된 연구인 [RAG](https://github.com/PARKYUNSU/pytorch_imple/tree/main/Agentic_RAG/Basic_Agentic_RAG)에 의해 다소 보완되었으나, RAG의 주요 문제는 검색(Retrieval) 단계에 있습니다. 이는 검색된 문서의 연관성에 크게 의존하게되어, 질문(Query)에 도움이 되지 않는 부정확하거나 불필요한 정보가 포함될 수 있기 때문입니다.

이에 본 논문은, 부정확한 검색 결과를 줄이고 답변의 견고성을 향상시키기 위해서 Corrective Retrieval Augmented Generation (CRAG)을 제안합니다.

CRAG는 내부 문서의 관련성 score를 기반으로 confidence를 계산해서, 문서들을 Correct, Incorrect, Ambiguous 세가지로 구분한 후, 각 파트에 맞게끔 문서를 정제하거나 외부 지식으로부터 보완한 최종 답변을 제공해서 RAG의 문제점을 보완합니다.

# 2. Related Work
## 2.1. Hallucinations of LLMs
LLM은 명령어 이해를 바탕으로 텍스트를 생성하지만, 여전히 가장 심각한 문제 중 하나는 Hallucinations(환각) 문제입니다.

부정확한 지식 및 과거 정보를 기반으로 문장을 이해 및 답변하는 모델은 Fine-tuning 및 재학습을 통해 더 정확한 정보를 재공함으로써 환각 현상을 막을 수 있습니다. 그러나 이 방법은 시간과 비용이 많이 들 수밖에 없습니다.

## 2.2. RAG (Retrieval-Augmented Generation)
기존 RAG는 LLM의 입력된 Query에 검색된 문서를 추가적인 정보원으로 재공함으로써 환각 문제를 완화하는 모델입니다.

하지만 앞서 설명한 RAG의 문제는 그 자체 검색(Retrieval)에 문제가 있으며 부정확한 검색은 답변 결과에 영향을 끼치기 마련입니다.

다음은 부정확한 Document로 인안 RAG의 잘 못된 답변을 생성하는 그림입니다.
<br/>
<br/>
<br/>
<img src="https://github.com/user-attachments/assets/ab8c0c94-1fd8-4ab5-854f-8d5de42804fb" width=400>
<br/>
<br/>
<br/>
RAG는 Retrieval $R$과 Generator $G$로 나뉩니다. 

입력 $X$와 $C = {d_1, …, d_N}$로 이루어진 대량의 Document에서 상위 $K$ 개의 문서 $D = {d_{r1}, …, d_{rk}}$를 검색해서 답변 $Y$를 생성하는 프로세스입니다. 이 과정을 수식으로 표현하면,

$$P(Y|X) = P(D|X)P(Y, D|X)$$

Retrieval과 Generator는 서로 영향을 긴밀하게 주고 있음을 보여줍니다, 즉, 검색이 실패하면 생성자가 아무리 뛰어나도 제대로 된 답변을 할 수 없음을 볼 수 있습니다.

## 2.3. Advanced RAG
또한, 최근 연구에서는 검색된 문서가 항상 정답을 보장하지 않을 수 있기 때문에, 어떤 경우에는 LLM 스스로 Retrieval 없이 답변하는 것이 더 정확할 수도 있다고 봅니다.

그래서 이런 자가 결정을 돕기 위해 Self-RAG 같은 접근법에서는 Critic Model 즉 평가자 모델을 도입해서, 검색을 할지 아니면 LLM이 스스로 답변을 할지 판단합니다.

# 3. CRAG
CRAG는 기존 접근법들과 다르게, 검색기능을 답변을 생성하는 보조 도구로 활용하거나 검색의 필요 여부에 집중하는 것이 아닌, 검색기가 부정확한 결과를 검색하는 상황 자체를 중점적으로 다루면서 RAG의 고질적인 문제를 해결하고자 설계되었습니다.

## 3.1. Overview of Model Inference
CRAG의 모델은 Retrieval, Knowledge Correction, Generation으로 나뉩니다.

Retreieval에서는 입력된 Query에 맞게 문서를 검색하고, 경량화된 Retrieval Evaluator(평가자)로 Query와 검색된 문서의 관련성 Score를 추정합니다.

이 Score는 총 3가지의 Cofidence로 정량화되어, 1) Correct, 2) Incorrect, 3) Ambiguous 로 나뉘어서 동작합니다.

#### 1) Correct
&nbsp; 검색된 문서들이 더 정밀한 Knowledge Strips로 정제되며, Decomposition, Filter, Recomposition 과정을 거쳐서 정제됩니다.

#### 2) Incorrect
&nbsp; 검색된 문서들을 사용하지 않고, Web Search로 검색된 정보를 사용합니다.

#### 3) Ambiguous
&nbsp; 검색된 문서가 부정확한 문서인지 아닌지 경정 내릴 수 없는 상태로, 이런 경우에는 두 가지 동작을 모두 사용하여 균형잡힌 검색 결과가 나오도록 유도합니다.

![image](https://github.com/user-attachments/assets/e080a3eb-971b-4310-9ca5-f3ed47cd6d6a)

## 3.2. Retrieval Evaluator
Retrieval Evaluator는 검색된 문서들이 실제로 질문에 대해 올바른 정보를 제공하는지 미리 확인하는 역할을 합니다. 그림에서 "Who was the screenwriter for Death of a Batman?" 질문에 대해 각 문서의 관련성을 평가하여, 올바르다면 "Correct", 그렇지 않다면 "Incorrect", 또는 불확실한 경우 "Ambiguous"로 분류합니다.

본 논문에서는 T5-Large 기반의 Retrieval Evaluator(약 0.77B 파라미터)를 미세조정하여, 기존 LLM(instruction-tuned LLaMA-2 7B)을 사용하는 critic model보다 낮은 비용으로 검색을 평가할 수 있습니다. 이렇게 계산된 Confidence는 최종적으로 검색 결과가 옳은지 판단하며 이후 Action Trigger(Correct, Incorrect, Ambiguous)를 결정합니다.

```text
Self-RAG에서 제공한 PopQA 데이터셋으로 Fine-tuning 합니다.

전체 PopQA 샘플 중 1,399개는 테스트용으로 분리하고, 나머지 샘플을 사용하여 T5-large 기반의 retrieval evaluator를 Fine-tuning 합니다.

이때, 정답(positive) 샘플에는 1, 부정(negative) 샘플에는 -1의 레이블을 할당하고, 학습 후 평가 시 각 문서의 관련성을 -1에서 1 사이의 점수로 산출하도록 모델을 튜닝합니다.
```

## 3.3. Action Trigger
CRAG에서는 검색된 문서의 Confidence에 따라 3가지로 동작합니다.

#### 1) Correct
&nbsp; 상한 임계값보다 높은 점수를 받은 문서는 Correct로 분류되어 관련성이 높고 신뢰할수 있는 지식으로 판단욉니다. Correct로 선택된 문서는 추가적으로, 문서 내 노이즈를 제가히기 위해서 Knowledge Strip을 추출 및 정제해서 사용해야합니다.

#### 2) Incorrect
&nbsp; 하한 임계값 이하인 문서들은 관련성이 없다고 판단되어 Incorrect로 분류 됩니다. 이 경우에는, 내부 문서를 이용해서 답변을 생성하지 않고 외부 Web Search로 답변을 생성합니다.

#### 3) Ambiguous
&nbsp; Correct와 Incorrect 중 어느 쪽으로도 명확하지 않은 경우에는 Retrieval Evaluator가 중간 점수로 문서들을 Ambiguous로 처리합니다. 이 경우 두 종류의 지식을 결합하여 상호 보완합니다.

CRAG의 초기 실험에는 Correct와 Incorrect만 사용할 경우 Retrieval Evaluator의 정확도 문제가 있었으나, Ambiguous을 도입함으로써 문제점을 완화할 수 있었습니다.

<img src="https://github.com/user-attachments/assets/bfa42608-ee26-486d-a0e6-2c23f4f52da3" widht=500>

## 3.4. Knowledge Refinement
검색된 문서에서 핵심 정보를 추출하기 위해서 decompose-then-recompose 방식을 사용합니다.

### Decompose
문서를 여러 개의 Internal Strips으로 나누어, 짧은 문장은 개별 Strip으로, 긴 문서는 몇 개의 문장 단위로 분할 합니다.

### Recompose
Fine-tuning된 Retrieval Evaluator를 통해 각 Strip의 관련성을 평가해서, 관련 없는 Strip은 제거한 후, 관련있는 Strip은 순서대로 이어 붙여서 최종 지식으로 사용합니다.

## 3.5. Web Search
CRAG에서는 내부 정보로만 답변을 잘 생성할 수 없음을 인지하면, 외부 지식을 활용하여 질문의 답변을 보완합니다.

입력 Query 문장을 ChatGPT를 이용하여 키워드 Query로 정제하고, API로 URL 링크를 생성해서 Wikipedia 등 공신력있는 웹페이제에서 정보를 우선적으로 검색합니다.


# 4. Experiment
본 실험에서는 CRAG의 RAG 기반 접근법이 단문 및 장문 생성 작업에서 얼마나 효과적인지, 그리고 다양한 실제 시나리오(단문 생성, 장문 생성, 참/거짓 질문, 객관식 질문)에서 일반성이 있는지를 평가하였습니다.

태스크 및 평가 데이터셋:
PopQA(단문 생성), Biography(장문 생성), PubHealth(참/거짓 질문), Arc-Challenge(객관식 질문)를 대상으로 평가를 진행하였으며, PopQA, PubHealth, Arc-Challenge는 정확도, Biography는 FactScore를 평가 지표로 사용하였습니다.

Baseline 비교:
CRAG는 검색을 활용한 표준 RAG 및 고급 RAG(Self-RAG, SAIL 등)와 비교되었고, 공개 LLM(예: LLaMA2-7B, Alpaca-7B, CoVE65B)과 propriety LLM(예: LLaMA2-chat13B, ChatGPT)도 평가에 포함되었습니다.



![image](https://github.com/user-attachments/assets/a21a45ea-d87d-4dbf-99bc-a3f212358a01)


# 5. Result
RAG 문제점 ~~~

CRAG로 RAG 프레임워크를 개선했지만 내부 지식만으로는 한계가 있어 외부 웹검생을 통한 보완이 불가피한 점을 한계점으로 꼽고있습니다.
