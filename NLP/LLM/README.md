# LLM (Large Language Model)

Large Language Model을 기본 Base에서 실험적인 방법으로 튜닝해서 설계하기

| Category           | Details                   |
|--------------------|---------------------------|
| GPU Model          | NVIDIA Tesla P100         |
| Architecture       | Pascal                    |
| CUDA Cores         | 3584                      |
| Memory Capacity    | 16 GB HBM2                |
| Memory Bandwidth   | ~732 GB/s                 |
| FP32 Performance   | ~10.6 TFLOPS              |
| Compute Capability | 6.0                       |


Base

| 항목               | 값                                                              |
|------------------|-----------------------------------------------------------------|
| **Dataset**         | Harry Potter and the Chamber of Secrets                         |
| **문자 수**        | 488,771                                                         |
| **토큰 수**        | 130,520                                                         |
| **총 에포크**      | 100                                                             |
| **에포크당 시간**   | 약 2분 21초 (약 141초)                                             |
| **총 학습 시간**     | 약 3.9 ~ 4시간                                                   |
| **초기 평균 손실**   | 약 4.42 (Epoch 1)                                               |
| **최종 평균 손실**   | 약 0.166 (Epoch 100)                                              |
