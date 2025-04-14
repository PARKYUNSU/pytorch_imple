# Transformer

"Attention Is All You Need"

-Ashish Vaswani, Noam Shazeer

[Read the Paper](https://arxiv.org/pdf/1706.03762)

---
## 논문 리뷰 영상

[![YouTube Video](https://img.youtube.com/vi/Tj5xxUjR_bQ/0.jpg)](https://www.youtube.com/watch?v=Tj5xxUjR_bQ)

---
 
# 1. Introduction

RNN, LSTM, gated recurrent 신경망은 언어 모델과 번역에 많이 사용되었습니다. 이 같은 순환 모델들은 보통 입출력 시퀀스의 각 심볼 위치에 맞춰 계산을 분할 합니다. 즉 연산 지접을 정렬하여, 이전 은닉 상태 $h_{t-1}$와 $t$ 시점 입력을 함수로 새로운 은닉 $h_{t}$를 생성합니다.

<img src="https://github.com/user-attachments/assets/0ba2486f-dbe2-43ed-98a9-fe8dadc41941" width=500 alt="RNN 표현방식">

| RNN 재귀적 표현(좌측), 펼친 표현(우측)

그러나, 이러한 구조적 특징으로 병렬적으로 학습하기 어렵고, 시퀀스의 길이가 길어질수록 메모리 제약 및 장기 의존성 문제로 긴 예제의 배치를 늘릴기엔 한계가 있습니다.

이러한 문제점을 해결 하기위해, 논문에서는 입출력 시퀀스에서 거리와 무관하게 모델링이 가능한 Attention을 이용해서 순환 구조를 배제하여 오직 Attention 메커니즘을 사용해서 병렬 처리 및 전연적인 정보를 출력하는 모델인 **Transformer** 모델을 제안합니다.

<p align="center"><img src="https://github.com/user-attachments/assets/0fe81c35-2817-4769-8f1e-fa6a03714d2b" width=400></p>
<p align="center">| Trnasformer Architecture</p>

# 2. Trnasformer

<img src="https://github.com/user-attachments/assets/97644d55-ac8a-491d-a548-fdbfd7f82c70" width=400>

| 간단한 Encoder-Decoder 구조

Transformer는 기존의 seq2seq 모델처럼 인코더-디코더 구조를 가지지만, RNN 없이 오직 어텐션 메커니즘만으로 시퀀스를 처리합니다.

인코더 디코더 각각 하나의 RNN이 시간축 $t$를 가지는 구조 → 인코더 디코더 각각 $N$개의 블록으로 구성


<img src="https://github.com/user-attachments/assets/ee1b7ff8-d704-4dca-aecc-0afecf092fed" width=400>

| 논문에서는 $N=6$으로 6개의 블록 구성

Trnasformer 모델 구조적 위해를 위해 "입력 부분" → "인코더 부분" → "디코더 부분" 으로 단계적으로 확대해서 설명하겠습니다.

## 2.1. Encoder - Decoder

<img src="https://github.com/user-attachments/assets/17ca3adb-ce76-4489-a29c-92fba694fe47" width=400>

| Encoder-Decoder 구조

위에서 서술한 것처럼 Trnasformer는 seq2seq 처럼 인코더 디코더 구조를 가지고 있으며, 인코더에서 정보를 받아 디코더가 결과를 생성합니다.

Trnasformer 디코더는 seq2seq 구조처럼 sos → eos 출력 까지 연산을 진행 합니다.


## 2.2. Encoder

<img src="https://github.com/user-attachments/assets/7aa222e3-bfea-4ad6-b088-7d504f4a5254" width=200>

| Transformer Encoder

### 2.2.1. Positional Encoding

기존 RNN에서는 단어를 순차적으로 받아서 처리하기 때문에 자연스럽게 단어의 위치정보(position information)를 확인할 수 있었습니다.

 - 순환구조로 앞의 연산이 끝나야 뒤의 연산을 진행할 수 있어서 계산할 유닛이 많아도 한 번에 1개씩 처리한다. 즉, 연산이 느리다.

그러나 Transformer는 단어를 순차적으로 받는 방식이 아니라 전체 시퀀스를 동시에 처리합니다. 이렇게 순서를 무시한 병렬 처리는 단어의 위치정보를 잃어버릴 수 있습니다.

 - Transformer는 압력된 문장을 병렬로 한 번에 처리하는 특징을 가지고 있다. 즉 연산이 빠르다.

논문에서는 이를 해결하기 위해 **Positional Encoding** 을 사용해서 입력 데이터에 순서 정보를 추가합니다.



그러나, [Language Modeling with Deep Transformers-2019](https://arxiv.org/pdf/1905.04226) 논문에서는 autoregressive 언어 모델에서 causal masking이 토큰 순서를 암묵적으로 반영하기 때문에, 깊은 Transformer가 다층 self-attention과 residual 연결을 통해 내재적 위치 정보를 학습할 수 있어 별도의 positional encoding이 필요 없다고 주장합니다.

<details>
<summary>Positional Encoding 톺아보기⁉️💡</summary>

 ### 2.2.1.1. Position Information

---

**"Not"** 의 위치로 인해 문장의 뜻이 달라진다.

### 1. Although I did **"not"** pass the test, I could get a scholarship.
  →  시험에 통과하지 못했지만, 장학금을 받을 수 있었다.

### 2. Although I did pass the test, I could **"not"** get a scholarship.
  → 시험에는 통과했지만, 장학금을 받을 수 없었다.

위치정보는 문장의 문맥을 이해하기에 중요한 정보로, Positional Encoding을 통해 위치정보를 더해줘야 한다.

<img src="https://github.com/user-attachments/assets/61efbf4a-76b3-4f70-b1e2-d89157c1f87e" width=600>

| Positional Encoding + 입력 단어 벡터

---

### 2.2.1.2. Positional Encoding Rules

Positional Encoding에는 2가지 규칙이 있습니다.

### 1) 모든 위치 값은 시퀀스의 길이나 Input에 관계없이 동일한 식별자를 가져야한다.
 시퀀스가 변경되어도 위치 임베딩은 동일하게 유지.

<img src="https://github.com/user-attachments/assets/44e00b49-f556-486f-b397-c75d74ab888f" width=600>


### 2) 모든 위치 값은 너무 크면 안된다.
 위치 값이 너무 커지면, 단어 간의 상관관계를 유추할 수 있는 의미정보가 커져 Attention Layer에서 학습이 안 될 수도 있다.

<img src="https://github.com/user-attachments/assets/b52aab36-4dd5-44c8-a031-6536e131a3b9" width=600>

---

### 2.2.1.3. Positional Encoding with Sine & Cosine 함수

Sine, Cosine 함수는 위의 2가지 규칙을 만족하면서 위치벡터를 추가할 수 있습니다.

### 1) 모든 위치 값은 너무 크면 안된다.
 - Sine & Cosine은 "-1 ~ 1" 사이를 반복하는 주기함수이다. 즉, 1을 초과하지 않으며, -1로 미만으로 가지 않으므로 두 번째 규칙을 지킬 수 있다.

  <img src="https://github.com/user-attachments/assets/e1b7785f-6bde-4324-aea6-86618b733a16" width=600>

### 2) 같은 위치의 토큰은 항상 같은 위치 벡터값을 가지고 있어야 한다.
- Sine &  Cosine은 주기함수로 -1 ~ 1 사이를 반복하므로 단어 토크들의 위치 벡터 값이 중복 될 수 있다.

   Sine 함수를 예를들어, 단어 토큰이 주어졌을 때, 1 번째 토큰(position 0)과 9 번째 토큰(position 8)의 경우 위치 벡터 값이 같아지는 문제가 발생한다.

  <img src="https://github.com/user-attachments/assets/bac65036-ac22-4eb5-a8a1-bad71b4d10db" width=600>

  하지만 Positional Encoding은 스칼라 값이 아닌 벡터 값으로 단어 벡터와 같은 차원을 지닌 벡터 값이다.

  <img src="https://github.com/user-attachments/assets/0c4d0964-9ced-4714-995c-9fee1f5ccb75" width=600>

  즉, 하나의 위치 벡터가 4개의 차원으로 표현된다면, 각 요소는 서로 다른 4개의 주기를 갖게 되어 서로 겹쳐지지 않는다. 고로 첫 번째 규칙을 지킬 수 있다.
</details>

<img src="https://github.com/user-attachments/assets/104f94c5-30bf-4439-88dc-2e423126040c" width=600>

| Position Encoding이 추가된 Encoder-Decoder

Positional Encoding 값은 아래의 두 함수를 사용합니다.


<p align="center">
$$
PE_{(pos, 2i)} = \sin\left({pos}/{10000^{{2i}/{d_{model}}}}\right)
$$
</p>

<p align="center">
$$
PE_{(pos, 2i+1)} = \cos\left({pos}/{10000^{{2i}/{d_{model}}}}\right)
$$
</p>

$Where:$
  - $pos$ : 입력 문장의 임베딩 벡터의 위치
  - $i$ : 임베딩 베터 내의 차원의 인덱스
  - $d_{model}$ : Transformer의 모든 층의 출력 차원 (Transformer Hyperparameter, 그림에서는 4)

<p align="left"><img src="https://github.com/user-attachments/assets/8bac064e-36cb-4da5-8492-0fff88be6bd3" width=600>
 <img src="https://github.com/user-attachments/assets/103574bc-89b0-4fc2-aa22-6bbd3a5e1ef7" width=400></p>

임베딩 벡터 내의 각 차원의 인데스가 짝수인 경우 $$(pos, 2i)$$에는 Sine 함수를 사용하고, 홀수인 경우 $$(pos, 2i + 1)$$ 에는 Cosine 함수를 사용


### 2.2.2. Attention

<img src="https://github.com/user-attachments/assets/16837e94-898c-48e9-a6de-cbb914890068" width=400>

Transforemr에서는 Attention이 어디서 이루어지는지에 따라 즉, 벡터의 출처가 어디냐에 따라 부르는 이름이 달라집니다.

```
인코더 Self Attention : Query = Key = Value
디코더 Masked Self Attention : Query = Key = Value
디코더 Encdoer-Decoder Attention : Query : 디코더 벡터 / Key = Value : 인코더 벡터
```

<img src="https://github.com/user-attachments/assets/2fa2e6d6-067c-4907-9433-9dc8a577a598" width=600>

### 2.2.2.1. Attention 함수

파이썬의 딕셔너리 자료형은 키(Key)와 값(Value)이라는 두 개의 쌍으로 구성되어, 키를 통해서 맵핑된 값을 찾아낼 수 있다는 특징이 있습니다.

```
# 파이썬의 딕셔너리 자료형
# 키(Key) : 값(value)의 형식으로 키와 값의 쌍(Pair)을 선언한다.
dict = {"A" : "Apple", "B" : "Banana"}
```
```
print(dict["A"]) # A 라는 키에 해당되는 값을 출력
Apple
```
```
print(dict["B"])  # B 라는 키에 해당되는 값을 출력
Banana
```

<img src="https://github.com/user-attachments/assets/f67af02e-ab70-4cb3-9454-907dede53596" width=400>

어텐션을 함수로 표현하면 주로 다음과 같이 표현됩니다.

$$Attention(Q, K, V) = Attention Value$$

어텐션 함수는 주어진 '쿼리(Query)'에 대해서 모든 '키(Key)'와의 유사도를 각각 구합니다. 그리고 구해낸 이 유사도를 키와 맵핑되어있는 각각의 '값(Value)'에 반영해줍니다. 

그리고 유사도가 반영된 '값(Value)'을 모두 더해서 리턴합니다. 여기서는 이를 어텐션 값(Attention Value)이라고 합니다.

<img src="https://github.com/user-attachments/assets/8bc41c7d-1ddf-4201-a2d6-6e843c4dff5c" width=400>

```
"The cat sat on the mat because it was soft and warm."
```

위의 예시 문장을 번역하면 "그 고양이는 매트에 앉았다, 왜냐하면 그것은 부드럽고 따뜻하기 떄문이다." 라는 의미입니다. 그러나 여기서 그것(It)에 해당하는 것이 cat 인지 mat인지 사람은 구분할 수 있지만, 기계는 그렇지 않습니다.

하지만 Attention에서는 입력 문장 내의 단어들 끼리 유사도를 구하므로, "it"이 참조하는 대상인 "mat"에 강한 어텐션 값을 가지게 됩니다.

또한 "soft"와 "warm" 같은 형용사들은 문맥상 "mat"와 연결될 수 있어 약한 어텐션 값을 가지게 됩니다.

### 2.2.2.2. Q, K, V 벡터 생성

Self Attention은 입력 문장의 단어 벡터로 부터 Q(Query) K(Key) V(Value) 벡터를 생성합니다.

각 단어 벡터는 초기 차원($d_{model}$)에서 더 작은 차원 ($d_k=d_v=64$) 로 변환합니다

단어 벡터에 학습 가능한 가중치 행렬($W_Q, W_K, W_V$)을 곱해서 Q, K, V 벡터를 생성합니다.

<img src="https://github.com/user-attachments/assets/70a15c72-4df7-4f04-94cd-96717cd243ea" width=400>

### 2.2.2.3. Scaled Dot Product Attention

Q, K, V 벡터를 생성후, 각 Q 벡터는 모든 K 벡터와 어텐션 스코어를 계산합니다.

스코어는 두 벡터 간의 내적을 사용하여, 이를 K 벡터 차원 $$\sqrt{d_k}$$으로 나눠 스케일링 합니다.

스코어에 Softmax를 취해 어텐션 분포를 구하고 이를 V 벡터와 가중합해 Attetion Value를 생성합니다.

<img src="https://github.com/user-attachments/assets/348e4ca4-208f-4afc-b229-c90b4e177993" width=600>

<img src="https://github.com/user-attachments/assets/085ebc7a-aef7-4893-8328-a09f48886914" width=600>

그러나 효율적인 연산을 위해 하나씩 Attention Value를 구하는 것이 아니라, 행렬 연산으로 처리합니다.

문장 전체 단어 벡터를 모아 문장 행렬 $$(seq-len, d_{model})$$을을 구성하고, 문장행렬에 학습 가능한 가중치 행렬 $W_Q, W_K, W_V$를 곱해 $Q, K, V$ 행렬을 생성합니다.

<img src="https://github.com/user-attachments/assets/2dda0632-412d-4b8c-98ed-082e42777cb8" width=500>

그리고 행렬 연산을 수행하는데,

$Q$와 $K^T$의행렬 곱으로 어텐션 스코어 행렬을 생성하고, Softmax를 적용해 어텐션 분포 행렬을 만든고, 어텐션 분포 행렬과 V 행렬 곱으로 최종 Attention Value $$(Seq-len, d_v)$$ 행렬을 만듭니다.

이 과정을 수식으로 표현하면 다음과 같습니다.

$$Attention(Q,K,V) = Softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

<img src="https://github.com/user-attachments/assets/84435fef-5049-4331-b30b-0a71bdef48d9" width=500>

### 2.2.2.4. Multi-head Attention

논문의 저자는 한 번의 어텐션을 하는 것보다 여러번의 어텐션을 병렬로 사용하는 것이 더 효과적이라고 판단하였습니다. 그래서 $d_{model}$의 차원을 $num-heads$개로 나누어 
$d_{model}/num-heads$의 차원을 가지는 Q, K, V에 대해서 $num-heads$개의 병렬 어텐션을 수행합니다.

여러 개의 어텐션 헤드를 병별로 수행하여 정보를 다양한 관점에서 수집이 가능해집니다.
```
ex) head 1 은 it ↔ mat 연관도를 강조하며, head 2 는 it ↔ soft 연관도를 강조합니다.
```

<img src="https://github.com/user-attachments/assets/b0d2374d-5471-48a6-a792-6a2e3663832e" width=500>

병렬 어텐션을 모두 수행하였다면 모든 어텐션 헤드를 연결(concatenate)합니다. 모두 연결된 어텐션 헤드 행렬의 크기는 $$(Seq-len, d_{model})$$ 가 됩니다.

<img src="https://github.com/user-attachments/assets/06709c5a-4679-4776-9815-2ecce92fa089" width=400>

병렬로 게산되어 합쳐진 행렬은 최종 가중치 행렬과 곱해서 최종 경과를 생성합니다.

<img src="https://github.com/user-attachments/assets/68acda6e-6d2a-40e5-bf6a-10c9a20308d0" width=500>


### 2.2.2.5. Position-Wise FFNN

인코더와 디코더의 공통 서브층으로 각 단어(벡터) 별로 독립적으로 작동하는 완전 연결 신경망입니다. Multi-head Attention 결과를 입력으로 받아 추가 변환을 수행합니다.

$$FFNN(x) = ReLU(xW_1 + b_1)W_2 + b_2$$

<img src="https://github.com/user-attachments/assets/ef7bc2c5-a74f-4430-ad49-396f1391ef8c" width=300>

FFNN은 어텐션 결과를 기반으로 단어별 독립적인 특징 변환을 수행하며, 이를 통해 모델의 표현력을 강화하고 더 복잡한 관계를 학습할 수 있도록 돕습니다. 동시에 입력 크기를 유지하여 다음 층에서 처리가 용이하도록 만듭니다.

### 2.2.2.6. Residual Connection

잔차 연결은 입력 $x$와 어떤 함수 $f(x)$의 출력을 더해서 새로운 값을 생성하는 구조입니다. 서브층의 입력과 출력을 더해서 학습을 안정화시키고, 기울기 소실 문제를 완화합니다.

트렌스포머에서는 서브층 마다 잔차 연결이 적용되어 출력을 이어지는 구조를 형성하며, 정보 손실을 방지하고, 학습 효과를 극대화합니다.

<img src="https://github.com/user-attachments/assets/e7181dbb-54de-4c39-affc-629c3cc90156" width=300>

### 2.2.2.7 Layer Normalization

LayerNorm은 활성화 배치의 각 항목의 평균과 분산을 계산하여 이를 사용해 데이터를 정규화하는 기법입니다.

기존 Batch Normalization 에서 Layer Normalization으로 변경하는데, Batch가 많아야지 적용이 잘되는 Batch Normalization인데, 자연어 데이터는 너무 많아 배치 크기를 줄이는 경우가 많습니다.

그로 인해 배치크기가 작아져 평균과 분산 추정이 부정확해져 성능이 저하 됩니다.

입력 데이터의 모양이 $[N,C,H,W]$ 일 때, 각 배치 내에서 $[C,H,W]$ 모양의 모든 요소에 대해 평균과 분산을 계산합니다.

배치 크기에 의존하지 않아 배치 정규화에서 생기는 문제를 해결하며, 평균과 분산 값을 따로 저장할 필요가 없습니다.

| **특징**           | **BatchNorm**                     | **LayerNorm**                 |
|--------------------|-----------------------------------|-------------------------------|
| **평균/분산 계산 기준** | 배치 단위                        | 각 샘플(feature 차원)         |
| **배치 크기 의존성** | 크면 안정적, 작으면 성능 저하      | 독립적, 소규모 배치에서도 안정 |
| **순차 데이터 처리**  | 비효율적                         | 적합                          |
| **추론 단계**       | 평균/분산 저장 필요              | 추가 저장 필요 없음            |

<img src="https://github.com/user-attachments/assets/40930afb-50c9-4a99-9fd2-5b630d39b8e3" width=600>


층 정규화 수식은 다음과 같습니다.
벡터 $h$의 평균 $μ$ 와 분산 $σ^2$ 을 계산한 후, 각 차원 $h_k$의 값을 아래 수식으로 정규화

여기서 $ϵ$은 분모가 0이 되는 것을 방지하기 위한 작은 값

$$x̂ᵢₖ = (xᵢₖ - μᵢ) / √{(σᵢ² + ε)}$$

감마와 베타

정규화된 값에 학습 가능한 파라미터 $γ$와 $β$를 적용하여 최종 정규화 값 계산

$$yk =γ\hat{h}_k+β$$

$γ$와 $β$는 초기값으로 각각 1과 0을 설정하며 학습을 통해 최적화합니다.


## 2.2. Decoder

<img src="https://github.com/user-attachments/assets/a92046da-6fef-4240-827e-6f2b0371c5b6" width=200>


디코더는 임베딩과 포지셔널 인코딩을 거친 입력 문장 행렬을 받습니다. 트랜스포머는 seq2seq처럼 교사 강요(Teacher Forcing)를 사용하여 학습됩니다.

즉, 디코더는 번역할 문장의 전체 입력(<sos> je suis étudiant)을 한 번에 받습니다.

그러나, 트랜스포머는 입력 문장 행렬을 한 번에 받으므로, 현재 시점의 단어 예측 시 미래 시점의 단어 정보까지 참고할 수 있는 문제가 발생합니다.

예를 들어, "suis"를 예측할 때, <sos> je뿐만 아니라 suis étudiant도 참고 가능해집니다.


### 2.2.1. Look-ahead Mask

디코더의 첫 번째 서브층인 멀티 헤드 셀프 어텐션에서 미래 단어를 참고하지 못하도록 마스킹 수행합니다.

자기 자신과 이전 단어만을 참조 가능하도록 제한을 둬서 단어 예측 과정에서 올바른 정보 흐름을 유지하는 중요한 역할을 합니다.


<img src="https://github.com/user-attachments/assets/688c6f09-15b8-43bb-a970-cdf11a7f5294" width=500>

<img src="https://github.com/user-attachments/assets/4531c300-3623-43f3-9ca1-f8649ee40526" width=250>


### 2.2.2. Encoder-Decoder Attention

<img src="https://github.com/user-attachments/assets/0621c51a-521e-4649-81cf-aaba96f5ad8a" width=300>

디코더의 두 번째 서브층은 인코더의 정보를 활용하여 입력과 출력 간의 관계를 모델링하며, 인코더-디코더 구조에서 중요한 연결 역할을 수행합니다.

Query가 디코더 행렬, Key가 인코더 행렬일 때, 어텐션 스코어 행렬을 구하는 과정은 다음과 같습니다.

<img src="https://github.com/user-attachments/assets/94bb7080-8ca7-42b8-ae1b-08be4545e590" width=500>

## Experiment

### Data Info

| 항목                      | 내용                                                          |
|-----------------------|------------------------------------------------------------------|
| `seq_len`            | 입력 시퀀스의 길이 (10)                                          |
| `vocab_size`         | 단어 집합의 크기 (100)                                          |
| `total_samples`      | 전체 샘플 수 (`train: 1000`, `val: 200`)                         |
| `batch_size`         | 배치 크기 (32)                                                  |
| 데이터 구성           | `src`(입력 시퀀스), `tgt`(출력 시퀀스)                          |
| 마스킹 종류           | Padding Mask, Look Ahead Mask                                   |


### 실험 설정

| 하이퍼파라미터        | 값                                                              |
|-----------------------|------------------------------------------------------------------|
| `num_layers`         | 2                                                               |
| `d_model`            | 128                                                             |
| `num_heads`          | 4                                                               |
| `d_ff`               | 256                                                             |
| `dropout`            | 0.1                                                             |
| `lr` (학습률)        | 1e-3                                                            |
| `epochs`             | 100                                                              |

---

### Result

<img src="https://github.com/user-attachments/assets/d38334e2-25fb-4c67-bc76-315a6ec5803e" width=600>

