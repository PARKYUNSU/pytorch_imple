# DeepSeek-R1

"Incentivizing Reasoning Capability in LLMs via Reinforcement Learning" - 2025

ㅡ DeepSeek-AI

[Read the Paper](https://arxiv.org/pdf/2501.12948)

---

# Introduction

딥시크 r1 3가지 정도의 결과를 발표,
1. deepseek r1o
2. 기존 openai 지도학습으로 파인튜닝으로 데이터 주입으로 대답을 더 잘하게하는데

deepseek r1o를 지도학습을 하지않고 강화학습으로 모델을 학습시켰다.
파인튜닝한 데이터가 없어서 2가지 문재점이 있는데, 가독성이 떨어지는 답변, 다른나라 언어로 질문을 할시에 cot가 영어로 하기에 답변이 영어로 나오는 경우기 있다.

강화학습 전에 Cold-start를 데이터를 주입해서 파인튜닝해서 deepseek r1이나왔다

openai o1이랑 성능이 비슷해서 각광받았다.

Qwem 외 Llama에 지식증류를 해서 작은 모델에서도 높은 성능을 보임을 알렸다.


기존 LLM을 먼저 살펴볼 필요가있다.

OpenAI o1의 COT

COT 는 답변 생성전에 여러단계를 거치면서 생각을 하는 단계라고 보면 된다

이 COT의 답변 길이를 길게 함으로써 o1의 논리 능력이 높아졌음

단점, 실제 배포하고 test time에서 이 모델의 성능을 강화시키는 방법이 아찍까지는 없었다.

3가지 고안 방법을 하고 있는데

1) process-based reward
2) 강화학습
3) monte carlo tree search 방법 도입

그래서 Deepseek 팀에서는 새로운 방법을 통해서 LLM 성능을 발전 시켰다

supervised 데이터 없이 자기 혼자서 생각함으로 모델을 발전시키는

Deepseekv3 base를 기반으로 GRPO로 강화했다.

벤치마크를 성능으 살펴보니 AIME 수학문제 데이터가 있는데 15.6 -> 71% 까지 성능

Majority voting 86.7% 까지 성능을 올렸습니다. Majority voting이란 LLM이 수학문제를 여러번 풀어보는 거라 생각하면 된다.
그래서 그중 가장 많은 답변을 선택하는 기법


가독성 떨어짐과 언어 믹싱을 보완하기위해 deepseek zero를 만들게된다.

강화학습만으로 만드는 cold-start 데이터셋을 넣어줌으로 파인튜닝하였다.

여기에 SFT 데이터를 더 주입하고 또 강화학습을 시킴

논문의 저자는 여기에서 그치는게 아니라 KD를 qwen과 Llam에 주입시켜서 작은 모델에서도 흘륭한 성능을 발휘하도록 했다.




