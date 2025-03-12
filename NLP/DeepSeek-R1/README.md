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
DeepSeek을 예기하기전 먼저 기존 LLM을 먼저 살펴볼 필요가 있습니다. 최근 몇 년간, LLM은 Post-Training (사후 훈련)을 통해 추론 성능을 크게 항상시켰습니다.

OpenAI-o1 시리즈는 사후훈련 중 하나인 Chain-of-Thought(COT)를 도입했는데, 이는 답변 생선 전에 여러 단계를 거치면서 모델이 스스로 생각하는 과정입니다. OpenAI-o1은 이러한 COT를 답변의 길이를 늘려 논리젓 사고 능력을 크게 향상시켰습니다. 그러나, COT의 단점으로 실제 배포하고 테스트하는 시점에서 모델 성능을 더 강화시키는 방법은 아직까지 명확하지 않습니다.

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
<details>
<summary>강화학습 및 Surrogate Loss / 펼치기</summary>

**강화학습(Reinforcement Learning, RL)** 은 Agent가 환경과 상호작용하며 보상을 최대화하는 방향으로 스스로 학습하는 방법입니다.

- **Environments**  
  Agent는 행동(액션)을 선택하여 환경에 영향을 주고, 환경은 이에 따라 보상과 다음 상태를 반환합니다.

- **Policy:**  
  Agnet가 주어진 상태에서 어떤 행동을 취할지를 결정하는 전략입니다.
  
  보통 확률 분포 $\pi_\theta(a \mid s)$로 표현됩니다.

- **Reward**  
  Agent가 특정 행동을 취한 결과로 받는 피드백으로, 보상을 최대화하는 방향으로 학습하게 됩니다.

- **Advantage**  
  특정 행동이 현재 상태에서 평균적인 기대 보상보다 얼마나 더 좋은지를 나타내는 값입니다.
  
  보상만으로 학습하면 절대 점수만 고려되지만, Advantage를 사용하면 "현재 수준에서 얼마나 개선되었는지"를 평가할 수 있습니다.

- **Policy 업데이트**  
  Agent는 수집한 경험(상태, 행동, 보상 등)을 기반으로 Policy을 업데이트합니다.
  
  업데이트는 보통 Policy 그래디언트 방법을 사용하여 여러 반복(iteration) 과정을 통해 점진적으로 개선됩니다.

---

**Policy 업데이트 & Surrogate Loss**

Agent는 수집한 경험을 바탕으로 Policy 파라미터 $\theta$ 를 업데이트합니다.

Policy 업데이트의 목표는 Old Policy와 New Policy 간의 행동 확률 비율을 고려하여 Advantage(Advantage)를 최대화하는 것입니다.

$$\max_{\theta} \left[\frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)} \hat{A}_t \right]$$

여기서 $$r_t(\theta)$$는 Old Policy과 이전 Policy의 행동 확률 비율이며, $\hat{A}_t$ 는 Advantage입니다.

</details>

### 3.1. Reinforcement Learning on the Base Model : PRO와 GRPO 비교

강화학습에서는 Policy 업데이트를 위해 보통 Actor-Critic 방식, 즉 Policy 모델과 Critic 모델을 함께 사용합니다. 대표적인 알고리즘인 PPO(Proximal Policy Optimization)는 2017년에 제안되었습니다.

Policy 업데이트 시 KL 발산을 이용해 신뢰 영역을 설정하여 Nwe Policy이 Old Policy와 크게 차이나지 않도록 보장하는 TRPO의 신뢰 영역 보장을 근사적으로 구현하면서 클리핑(clipping) 기법을 도입해 Policy 업데이트의 안정성을 확보합니다.



### TRPO(Trust Region Policy Optimization) Process

$$\textbf{TRPO:} \quad \max_{\theta} \ \mathbb{E}t \Bigg[
\frac{\pi_{\theta}(a_t \mid s_t)}{\pi_{\text{old}}(a_t \mid s_t)} \\hat{A}t
\-\
\beta \mathrm{KL}\Big[\pi_{\text{old}}(\cdot \mid s_t)∥ \pi_{\theta}(\cdot \mid s_t)\Big]
\Bigg]$$

TRPO는 objective term $\frac{\pi_{\theta}(a_t \mid s_t)}{\pi_{\text{old}}(a_t \mid s_t)} \\hat{A}t$ 을 최대화하면서 penalty term $\mathrm{KL}[\pi_{\text{old}}]$를 최소화하는 것을 목표로 합니다.

즉, Policy의 improvement step을 최대한 크게 가져가면서, 동시에 penalty term은 old policy와 new policy의 차이가 너무 크게 변경되지 않도록 KL divergence를 통해 억제하는 것 입니다.


### PPO(Proximal Policy Optimization) Process

$$\textbf{PPO:} \quad \max_{\theta} \ \mathbb{E}t \Bigg[
\min\Bigg(
\frac{\pi_{\theta}(a_t \mid s_t)}{\pi_{\text{old}}(a_t \mid s_t)}\hat{A}t,
\mathrm{clip}\Bigg(
\frac{\pi_{\theta}(a_t \mid s_t)}{\pi_{\text{old}}(a_t \mid s_t)}\,
1-\epsilon\,1+\epsilon
\Bigg)\hat{A}_t
\Bigg)
\Bigg]$$

PRO에서 objective term $$\frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{old}}(a_t \mid s_t)}$$을 최대화 하는 부분은 TRPO와 동일하며, penalty term에서 KL divergence을 사용하는 대신 clipping 방식을 사용했습니다. 이를 통해 second-order method가 아닌 first-order method로 계산합니다.

$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{old}}(a_t \mid s_t)}$은 특정 action을 취할 old policy와 new policy의 행동 확률 비율입니다.

이 비율이 1 (즉, 변화 없음)에서 크게 벗어나지 않도록, Clippling은 $r_t(\theta)$ 를 $[1-\epsilon,\, 1+\epsilon]$ 범위로 제한하여, policy가 과도한 업데이되는 것을 방지합니다.

### Algorithm 1: PPO (Actor-Critic Style)
```rust
for iteration = 1, 2, ... do # 학습과정 반복, 각 반복에서 Policy 업데이트
    for actor = 1, 2, ..., N do # 병렬 actors를 통해 데이터 모으기
        Run policy π_old in environment for T timesteps $ 각 actor는 old policy를 사용해서 T step 동안 시뮬레이션 진행
        Compute advantage estimates A1, ..., A_T # state, action, reward를 수집하고 advantage를 계산
    end for
    Optimize surrogate L wrt θ, with K epochs and minibatch size M ≤ N*T # 수집된
    π_old ← π
end for
```
#### PRO의 장단점

- 장점: 구현과 튜닝이 간단하며, 안정적인 학습을 보장합니다.
  
- 단점: on-policy 방식으로 매 업데이트마다 새로운 데이터를 생성해야 하므로 샘플 효율이 낮을 수 있습니다.

### GRPO(Group Relative Policy Optimization) Process
<img src="https://latex.codecogs.com/png.image?\inline&space;\dpi{110}\bg{white}$$J_{GRPO}(\theta)=\mathbb{E}_{q\sim&space;P(Q)\{o_i\}_{i=1}^G\sim\pi_{\theta_{\text{old}}}(O|q)}\left[\frac{1}{G}\sum_{i=1}^G\left(\min\left(\frac{\pi_{\theta}(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)}A_i,\text{clip}\left(\frac{\pi_{\theta}(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)},1-\epsilon,1&plus;\epsilon\right)A_i\right)-\beta&space;D_{KL}(\pi_{\theta}\|\pi_{\text{ref}})\right)\right]$$" title="$$J_{GRPO}(\theta)=\mathbb{E}_{q\sim P(Q)\{o_i\}_{i=1}^G\sim\pi_{\theta_{\text{old}}}(O|q)}\left[\frac{1}{G}\sum_{i=1}^G\left(\min\left(\frac{\pi_{\theta}(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)}A_i,\text{clip}\left(\frac{\pi_{\theta}(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)},1-\epsilon,1+\epsilon\right)A_i\right)-\beta D_{KL}(\pi_{\theta}\|\pi_{\text{ref}})\right)\right]$$" />

DeepSeek에서는 GRPO를 사용했으며, GRPO는 2023년경 제안된 새로운 강화학습 알고리즘으로, 특히 대형 언어 모델(LLM)의 추론, 수학 문제 해결 등 고급 능력을 강화시키기 위해 설계되었습니다.

PPO와 달리 별도의 가치망(critic)을 두지 않고, 한 프롬프트에 대해 Old Policy $π$를 이용해 여러 응답(출력)을 생성합니다.

생성된 $k$개의 응답 각각에 보상 점수를 부여한 후, 그룹 내 보상 점수들의 평균이나 최고 점수를 기준으로 각 응답의 Advantage을 계산합니다. 이를 통해, 그룹 내에서 상대적으로 더 나은 응답의 생성 확률을 높이고, 열등한 응답은 줄이도록 Policy을 업데이트합니다.

```text
시험 상황에서, 단순히 절대 점수만 보는 대신, 각 학생의 평소 기대치와 비교해 개선된 정도에 따라 보상을 주는 것과 유사합니다.
```

#### GRPO 장단점
- 장점
   - Critic 모델이 없으므로 메모리와 연산 비용이 크게 절감됩니다.
   - 그룹 내 보상 점수의 상대적 차이로 Advantage를 계산하므로, 복잡한 가치 예측 네트워크가 필요 없습니다.
   - 초기 실험 결과, GRPO를 적용한 모델은 PPO 기반 모델보다 특정 작업(예: 수학적 추론)에서 더 높은 성능을 보였습니다.
- 단점
   - Reward 함수 설계가 매우 중요하며, 이를 부정확하게 설계할 경우 잘못된 방향으로 학습될 위험이 있습니다.
   - 한 프롬프트 당 여러 응답을 생성해야 하므로, 대규모 샘플이 필요하며 하이퍼파라미터 튜닝이 요구됩니다.
   - 새로운 방법이기 때문에 수렴 보장이나 안정성에 관한 이론적 분석이 아직 부족합니다.

### 2. Reward Modeling

### 3. Training Template

### 4. Performance, Self-evolution Process and Aha Moment of DeepSeek-R1-Zero


