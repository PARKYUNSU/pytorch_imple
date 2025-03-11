# DeepSeek-R1

"Incentivizing Reasoning Capability in LLMs via Reinforcement Learning" - 2025

ㅡ DeepSeek-AI

[Read the Paper](https://arxiv.org/pdf/2501.12948)

---

# 1. Abstract
DeepSeek Research는 DeepSeek-R1 발표에 있어 3가지의 결과를 논문에서 소개합니다.

### 1.1. DeepSeek-R1-Zero
기존의 LLM은 Supervised Learning을 통한 Fine-Tuning으로 데이터를 추가 학습하면서 추론 능력을 항상시켰습니다. 이에 반해, DeepSeek Researcher들은 DeepSeel-R1-Zero가 Supervised Learning 없이 순수 Reinforcement Learning만으로 학습되어도 강력한 추론 능력을 발휘할 수 있음을 보여줍니다.

그러나 Fine-Tunning한 데이터가 없어서 DeepSeek-R1-Zero는 2가지의 문제점이 있는데, 첫째는 Poor Readability(가독성이 떨어지는) 답변, 다른 하나는 영어와 중국어 외 언어로 질문을 할시에 CoT(Chain of Thought)가 영어로 작성되기에 답변도 영어로 나오는 Language Mixing 문제점이 있습니다.

### 1.2. DeepSeek-R1
그래서 Reinforcement Learning 전에 Cold-start를 데이터를 주입해서 Fine-Tuning을 진행하여, DeepSeek-R1-Zero의 문제점을 개선하고 OpenAI-o1-1217과 동등한 추론 능력을 달성하여 각광을 받았습니다.

### 1.3. Knowledge Distrilation
DeepSeek 연구진들은 또한 연구 커뮤니티에 추가적으로 지원하기 위해, DeepSeek-R1을 Distrilation 하여 Qwen과 Llama에 도입함으로 6개의 증류된 오픈소스(1.5B, 7B, 8B, 14B, 32B, 70B)와 DeepSeek-R1-Zero와 DeepSeek-R1을 제공하고 있습니다.

# 2. Introduction
DeepSeek을 예기하기전 먼저 기존 LLM을 먼저 살펴볼 필요가 있습니다. 최근 몇 년간, LLM은 Post-Training (사후 훈련)을 통해 추론 성능을 크게 항상시켰습니다. 사후훈련의 예를 들어, OpenAI-o1 시리즈는 Chain-of-Thought(COT)를 도입했는데, 이는 답변 생선 전에 여러 단계를 거치면서 모델이 스스로 생각하는 과정입니다. OpenAI-o1은 이러한 COT를 답변의 길이를 늘려 논리젓 사고 능력을 크게 향상시켰습니다. 그러나, COT의 단점으로 실제 배포하고 테스트하는 시점에서 모델 성능을 더 강화시키는 방법은 아직까지 명확하지 않습니다.

이러한 단점을 해결하기 위해서 기존 연구에서는 3가지 고안 방법을 진행했습니다.

#### 1) Process-based Reward
#### 2) Reinforcement Learning
#### 3) Monte Carlo Tree Search

그러나 이 방법들도 OpenAI o1 시리즈와 비교할 만한 일반적인 추론 성능 향상을 이루지는 못했습니다. 이에 DeepSeek Researcher들은 Supervised Learning 데이터 없이 자기 혼자서 생각하면서 모델을 발전시키는 방법을 개발하게 되었습니다. 

논문 저자들은 DeepSeek-V3-Base를 기본 모델로 GRPO (Group Relative Policy Optimization)를 아용해 강화시켰습니다. 훈련을 통해서 DeepSeek-R1-Zero 모델이 자연스럽게 만들어졌으며, 벤치마크 성능을 살펴본 결과 수학문제 데이터인 AIME의 결과가 pass@1 Score가 15.6%에서 71% 까지 증가 했습니다.
또한, Majority voting 86.7% 까지 성능을 올렸습니다.
```text
pass@1

LLM이 질문 1개에 대해 1개의 code를 생성한 뒤에 Test Case 통과 여부를 점수로 계산
```

```text
Majority Voting

 LLM이 수학문제를 여러번 풀어 문제에 대해 생성된 솔루션들 중에서 가장 빈도가 높은 답변을 최종 선택하는 방식
```
<img src="https://github.com/user-attachments/assets/f98f3608-cf9e-4d7a-bb98-964c90d2648f" width=500>

| AIME Dataset Example

그러나 DeepSeek-R1-Zero는 Poor Readability와 Language Mixing의 문제가 있어서, 그 문제를 보완하기위해 Cold-Start Data를 도입한 DeepSeek-R1을 만들었습니다.

### DeepSeek-R1 Training Process

<img src="https://github.com/user-attachments/assets/55d8d2f9-1cc5-4eba-9100-16d2683eac04" width=500>

# 3. Approach

### 3.1. GRPO(Group Relative Policy Optimization)


### 2. Reward Modeling

### 3. Training Template

### 4. Performance, Self-evolution Process and Aha Moment of DeepSeek-R1-Zero


