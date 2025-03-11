# CLIP

'Learning Transferable Visual Models From Natural Language Supervision' - 2021

ㅡ Alec Radford, Jong Wook Kim, Chris Hallacy

[Paper to Read](https://arxiv.org/pdf/2103.00020)

---

# Introduction

CLIP은 기존의 CV Model과 NLP Model의 한계를 넘어서려는 새로운 시도를 담고 있어, 2021년 당시 많은 주목을 받았습니다. GPT-4, LLaVA 등 다양한 Large Multimodal Model (LMM)의 시초 연구라고 할 수 있는 모델입니다.

기존에는 CV Model은 주로 이미지만을 학습하여 그 성능을 향상시켜왔습니다. 하지만 이러한 접근 방식은 모델의 일반화 능력에 한계가 있습니다. 반면, NLP 모델들은 대규모 언어 데이터를 학습함으로써 급격한 발전을 해왔습니다.
CLIP의 저자들은 이러한 배경에서 언어 모델처럼 대규모 데이터셋을 학습하는 방식이 Image Recognization 분야에서도 중요한 역하을 할 수 있다고 생각했습니다. 이러한 생각은 기존의 접근 방식에 대한 시각을 제공했고, CV 인공지능 분야에서 발전 가능성을 마련해주었습니다.

# 2. Approach

## 2.1. Computer Vision Model

기존 Computer Vision Model에서는 다양한 기법들을 사용해서 그 성능을 향상시켜왔습니다.

#### 1) Resnet, Inception 등을 활용한 깊은 모델 구조

#### 2) SENet, BAM, CBAM 등 Ateention 모듈을 확용한 모델

#### 3) Vision Transformer, ImageGPT 등 Transformer 구조를 활용한 모델

### Limitations

&nbsp; 그러나 이런 모델은 이미지만 학습한 모델로 일반화 능력이 부족학, 노이즈나 다른 도메인에는 그 성능이 좋지 못한 경우도 있습니다. 또한 사람이 라벨링한 ImageNet 데이터셋을 사용하기에 한계가 있습니다.

## 2.2. NLP Model

#### 1) Seq2Seq의 방식은 장기의존 기억 한계로 발전이 더뎠습니다.

#### 2) GPT시리즈(GPT-1, GPT-2, GPT-3, BERT) 등 LLM으로 긴 문장 처리 및 대규모 데이터 활용이 가능해졌습니다.

### Implications

&nbsp; 모델 규모가 커지고 데이터의 양도 기하급수적으로 증가했으며, 학습데이터가 많을 수록 성능이 계속적으로 개선되었습니다.

## 2.3. CLIP의 Background

논문의 저자는 2.1. CV의 한계점과 2.2. Nlp의 시사점의 방향을 가지고 한 단계 더 발전 가능성에 집중했습니다. 모델 규모가 큰 모델과, 기하 급수적으로 많은 큰 데이터를 Computer Vision을 사용하는 방식으로 한계를 극복하고자 했습니다.

# 3. CILP

논문의 저자는 데이터셋의 한계를 극복하기 위해 대용량의 Image-Text 쌍을 이루는 데이터를 이용하는 방식을 선택했습니다. 또한 기존 Vision Trasformer Model로는 학습이 어려워, Zero Shot Prediction 원리로 효과적인 학습을 진행했습니다.

## 3.1. NLP Supervision Learning

기존에 ImageNet에서 사람이 달았던 라벨을 사용하는 것이 아니라, 인터넷에서 이미지를 대거 수집하여, 이미지에 포함된 설명 문장을 그대로 사용합니다. 약 4억장(Image-Text pairs)의 이미지와 자연어 문장을 그대로 사용하는 아이디어입니다.

<br/>
ex) Image-Text pairs Dataset
<br/>

<img src="https://github.com/user-attachments/assets/5625ead9-61f9-44f2-aad9-09194f51ef3e" width=700>

## 3.2. CLIP Learning

ImageNet 처럼 CrossEntropy Loss를 사용하지 않고 Self-supervised Learing에서 사용한 [Contrastive Learning](https://github.com/PARKYUNSU/pytorch_imple/tree/main/CV/Self%20Supervised/SimSiam)을 사용합니다.

CLIP에서는 정답 쌍(i번째 Image-Text)은 Cosine Similarity 를 높이고 나머지는 낮춰서 대조적으로 학습합니다.

<br/>
<img src="https://github.com/user-attachments/assets/dd567125-77ed-4447-9296-ea4c20a7d17b" width=700>

## 3.3. Encoder

CLIP에서는 Image와 Text를 학습하기 위해서 벡터화를 진행해야합니다. 각 항목의 벡터화는 서로 다른 Encoder로 변환해서 사용합니다.

#### Image Encoder

&nbsp; 마지막 Pooling을 Attention Pooling으로 변경한 5가지의 ResNet 모델과 Vision Trnasformer 를 사용합니다.

#### Text Encoder

&nbsp; Transformer Encoder를 사용하며, 마지막 Token에서 추출된 Feature를 선형 변환하여 Image Feature와 같은 차원으로 맞춰서 사용합니다.

<img src="https://github.com/user-attachments/assets/d4c95df3-ed13-4bc1-99b7-14c7e678fbb0" width=500>

## 3.4. Zero-Shot

CLIP의 가장 큰 특징 중 하나로, Zero-SHot Prediction이 가능하다는 점 입니다. 기존 모델은 CV에서는 정해진 클래스(CIFAR-10 10개, ImageNet 1000개)만 예측 가능했으나, CLIP은 Image-Text Pairs Dataset으로 결합 학습해서, 새로운 클래스에 대한 Text 설명만 있으면 예측이 가능합니다.

CLIP에서 Zero-Shot은 다음과 같이 작동합니다.

#### 1) Image Feature 추출

&nbsp; 사전 학습된 CLIP의 Image Encoder로 이미지 Embedding 추출

#### 2) Class Feature

&nbsp; 분류하고자 하는 Class가 N개일 때, 각각 "A Photo of a {object}"로 문장을 만들고, CLIP의 Text Encdoer로 Embedding 추출

#### 3) Cosine Similarity

&nbsp; Image Embedding과 각 Class Embedding을 모두 비교해서, 가장 유사도가 높은 크래스를 예측값으로 결정합니다. 이를 통해서 Class 개수가 고정되지 않은 Zero-Shot Prediction이 가능해집니다.

<img src="https://github.com/user-attachments/assets/a1c3aefe-7bdd-484a-aca1-b7745495227b" width=700>

# 4. Experiments

CLIP을 다른 실험 비교군과 비교해보면서 CLIP의 성능을 알아봅니다.

## 4.1. Zero-Shot Transfer

### 4.1.1. Initial Comparison to Visual N-grams

Visual N-grams는 이미지 내에서 자주 함께 등장하는 시각적 패턴을 추출하고 이를 여러 단계로 확장해서 n-grams 형태로 구성함으로써 이미지 분류 또는 객체 인식을 수행하는 방법입니다.

실험 결과, 이미지 분류 문제에서 기존의 Visual N-grams 방식에 비해 CLIP은 다음의 3개의 데이터셋에서 모두 훨씬 뛰어난 Zere-Shot 성능을 보였습니다.

<img src="https://github.com/user-attachments/assets/f29d1075-ccbd-4d88-b439-c1dc34917a33" width=400>

### 4.1.2. Prompt Engineering and Ensembling

대부분의 이미지 데이터셋은 클래스 이름만 존재하며, Context가 부족해서 Class Name의 Ambiguity(다의성)가 문제가 될 수 있습니다. CLIP에서는 GPT-3와 비슷하게 다양한 Prompt Text를 적절히 선택해 주면 분류의 성능을 더 높일 수 있습니다.

-   "A photo of a {label}, a type of pet."

-   "a satellite photo of a {label}."

-   "A photo of a big {label}."

-   "A photo of a small {label}."

이런 방식으로 바꾸면 분류 성능을 높일 수 있습니다.

<img src="https://github.com/user-attachments/assets/0800c19b-ed5d-441c-9896-d9c8839aa765" width=500>

### 4.1.3. Analysis of Zero-Shot CLIP Performance

27개 데이터셋 중 16개 데이터셋이 ResNet-50 Baseline 보다 우수한 성능을 보여줍니다. 데이터셋 특성에 따라 성능의 차이가 있습니다.
<img src="https://github.com/user-attachments/assets/542d5061-9c32-4020-bf2f-5ddc65ab9bbf" width=500>

또한 다른 모델의 Few-Shot 성능보다 CLIP의 Zero-Shot 성능이 더 나았음을 보여줍니다.
<img src="https://github.com/user-attachments/assets/a837bab0-eea2-4fb9-b037-cd7a02f33091" width=500>

### 4.1.4. Representation Learning

#### Representation Learning

Image의 특성을 최대한 잘 설명하는 Feature(Representation)을 뽑아서 이를 Downstream Task에 활용하는 학습으로, 얼마나 일반적이고 유용한 Representation을 얻는가를 중점으로 봅니다.

CLIP에서는 Image와 Text를 한꺼번에 학습함으로, 다양한 Text와 짝지어진 Image의 Representation을 학습했습니다. 이렇게 학습된 Image Encdoer는 Image만 주어져도 상당히 일반적이고 풍부한 시각적 Representation을 만들 수 있습니다.

<img src="https://github.com/user-attachments/assets/d3bdd478-741f-42ab-b037-91fa93dd3d19" width=700>

위의 그리멩서 CLIP(ResNet-50) 작은 모델은 기준 SOTA모델 보다는 약간 낮은 성능을 보였으나, 큰 모델(ResNet x64, ViT-B/32)은 대표 모델인 EfficientNet을 웃도는 성능을 보여줬습니다.

### 4.1.5. Comparison to Human Performance

CLIP의 성능을 또한 사람과 비교한 실험을 진행했습니다.
사람에게도 Zero-Shot 즉, 사전 정보 없이 재공하며, 또한 One-Shot, Two-Shot으로 학습을 평가합니다.

### 실험

#### 1) 총 5명에게 Oxford IIT Pets (개·고양이 37종)의 Test Split(3669장) Image를 차례로 보여주고, 해당 이미지가 어떤 종인지 맟추는 실험을 진행

#### 2) Zero-Shot에서는 기본적인 종 이름만 제공하고, 별도 예시 이미지를 보여주지 않게 실험

#### 3) One-Shot, Two-Shot에서는 각 종에 대해 1-2장의 예시 이미지를 추가로 제시해서 실험

### Result

CLIP의 분류 정확도가 인간보다 더 높게 나타났으며, 사람에게 예시 이미지를 더 보여줘도 CLIP의 성능을 따라 잡지 못한 경우가 많았습니다.

또한 CLIP이 틀린 사례의 이미지는 사람도 보기 어렵거나 헷갈리는 케이스가 많았습니다.

<img src="https://github.com/user-attachments/assets/300924e0-c352-44e2-9cc0-79e0f54ef5ba" width=500>

5. Broader Impacts
   CLIP의 훌륭한 성능은 어떻게 적용하느냐에 따라 광범위한 기능을 가질 수 있습니다.

CLIP은 OCR을 수행할 수 있고, 스캔한 문서를 검색가능하게 만들거나, 화면 판독 및 동장인식, 물체 분류, 얼굴 감정 인식 등 다양한 분야에 쓰일 수 있습니다.

그러나 CLIP이 학습한 데이터셋 내부에 사회적 편견이 그대로 녹아 들어갈 수 있고, 이를 학습한 모델 역시 그 성향이 비칠 수 있습니다.

5.1. Bias
모델이 훈련되는 과정에서, 그 데이터셋에 포함된 사회적·인종적·성별 편견이 그대로 학습될 수 있습니다.

ex)

-   FairFace 데이터셋에 ‘동물’, ‘범죄자’ 등 부정적 class를 추가했을 때, 특정 인종(‘흑인’) 및 남성 그룹에 대한 오분류율이 더 높게 나타났습니다.
-   국회의원 이미지 실험에서도, 남성은 범죄 관련 class와, 여성은 가사 관련 직업과 더 쉽게 연관지어졌습니다.

이로인해 모델의 오분류가 특정 집단(소수자·청소년·특정 연령·성별)에 더 크게 영향을 줄 수 있음으로, 차별적이거나 공격적인 레이블을 만드는 위험에 대해 해결방안이 필요합니다.

### 해결방안

모델 설계 및 데이터 구성 단계에서 편향을 인지하고 통제해야 합니다.

학습 과정에서 편향을 줄이는 알고리즘, 데이터 보정 기법, 그리고 테스트 시 bias 검증 프로토콜 등을 적용하는 것이 중요하기에, class name이나 label design 설계할때 주의가 필요합니다.

# 6. Conclusion

대규모의 Image-Test Pairs 데이터셋을 활용한 CLIP은 기존 Coputer Vision 모델의 한계를 극복하고 Zero-Shot 예측을 비롯한 강력한 일반화화 성능을 보여주었습니다.

이를통해 Multimodal 시대가 가속화되었지만, 동시에 사회적 편향과 윤리적 문제가 노출될 위험이 커졌습니다. Web Data에 내재된 편향이나 개인정보 침해, 감시 등에 활용될 소지가 있어, 안전장치 마련과 책임 있는 사용이 필수입니다. 그럼에도 불구하고 CLIP은 NLP-CV 정보 결합으로 Computer Vision 분야에서 새로운 가능성을 열었다는 점에서 큰 의의를 지닙니다.
