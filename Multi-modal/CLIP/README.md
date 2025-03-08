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
마지막 Pooling을 Attention Pooling으로 변경한 5가지의 ResNet 모델과 Vision Trnasformer 를 사용합니다.

#### Text Encoder

Transformer Encoder를 사용하며, 마지막 Token에서 추출된 Feature를 선형 변환하여 Image Feature와 같은 차원으로 맞춰서 사용합니다.

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
