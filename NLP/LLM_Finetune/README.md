# LLM Finetuning

[카카오 나노 2.1b 베이스 모델](https://huggingface.co/kakaocorp/kanana-nano-2.1b-base)을 활용해서 최소한의 예제를 바탕으로 Fine-Tuning 하는 코드입니다.


## 🖥️ 실험 환경 정리
| 항목                            | 내용                                             |
|---------------------------------|--------------------------------------------------|
| **Platform**           | Kaggle                                           |
| **Hardware**         | TPU VM v3-8 (8 TPU cores, 128 GB HBM)            |
| **OS**              | Ubuntu Linux 기반                                |
| **Frame work**                  | PyTorch, Hugging Face Transformers |
| **Model**                   | `kakaocorp/kanana-nano-2.1b-base`                |
| **Optimizer** | AdamW                                            |
| **Learning Rate**      | `1e-5`                                           |
| **Weight Decay**  | `0.01`                                           |
| **Batch Size**    | `2`                                              |
| **Epochs**            | `10`                                             |

## Training Epoch Plot
<img src="https://github.com/user-attachments/assets/25ff0a33-3a11-48ab-be7d-3aca04000d88" width=400>


## Dataset
QA의 형태의 Dataset으로 "|" 이 표시를 기준으로 Question Answer 쌍의 데이터 셋

[data/qa_customdata](https://github.com/PARKYUNSU/pytorch_imple/blob/main/NLP/LLM_Finetune/data/qa_customdata.txt)
```text
박윤수의 행운의 숫자는?|박윤수는 3과 7을 좋아합니다.
박윤수가 좋아하는 디저트는 무엇인가요?|박윤수는 티라미수와 마카롱을 좋아합니다.
박윤수가 즐겨 하는 게임은?|박윤수는 최근에 엘든 링에 빠져 있습니다.
박윤수가 자주 방문하는 도시는 어디인가요?|박윤수는 제주도를 좋아해서 자주 방문합니다.
박윤수의 취미 생활은?|박윤수는 사진 촬영과 하이킹을 즐깁니다.
박윤수가 가장 좋아하는 계절은?|박윤수는 가을을 가장 좋아합니다.
박윤수의 특별한 재능이 있나요?|박윤수는 그림을 잘 그립니다.
박윤수가 좋아하는 음악 장르는?|박윤수는 재즈와 힙합을 자주 듣습니다.
박윤수가 가장 좋아하는 색깔은?|박윤수는 하늘색을 가장 좋아합니다.
박윤수가 좋아하는 영화 장르는?|박윤수는 로맨틱 코미디 영화를 좋아합니다.
박윤수가 좋아하는 운동은 무엇인가요?|박윤수는 주말마다 사이클링을 합니다.
박윤수가 좋아하는 동물은?|박윤수는 고양이를 아주 좋아합니다.
박윤수가 주로 사용하는 소셜 미디어 플랫폼은?|박윤수는 인스타그램을 주로 사용합니다.
박윤수가 가장 좋아하는 음식은?|박윤수는 초밥과 라멘을 좋아합니다.
박윤수가 최근 본 드라마는 무엇인가요?|박윤수는 최근 '무빙'을 재미있게 봤습니다.
박윤수가 싫어하는 음식은 무엇인가요?|박윤수는 향이 강한 음식을 싫어합니다.
```

## Fine Tuning 전 답변
<img src="https://github.com/user-attachments/assets/86238ab2-b0e3-4570-9261-0fc7055c7728" width=600>

## Fine Tuning 후 답변
<img src="https://github.com/user-attachments/assets/9c08ec9f-8a53-4a00-a35e-2e142f6df15d" width=600>
