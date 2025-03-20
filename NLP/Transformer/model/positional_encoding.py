import torch
from torch import nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_seq_len):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # (max_seq_len, d_model) 형태의 zero 텐서 준비
        pe = torch.zeros(max_seq_len, d_model)  # (max_seq_len, d_model)

        # 위치 인덱스: [0, 1, 2, ..., max_seq_len-1]
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)  # (max_seq_len, 1)

        # 분수 항 (2i에 대해 사용)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # 짝수 인덱스(2i)에 대해 sin, 홀수 인덱스(2i+1)에 대해 cos 적용
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # (1, max_seq_len, d_model) 형태로 reshape해서 batch 차원과 브로드캐스팅 가능하게
        pe = pe.unsqueeze(0)  # (1, max_seq_len, d_model)

        # 학습되지 않는 버퍼로 등록 (매개변수 X)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 입력 시퀀스 길이에 맞게 슬라이싱
        seq_len = x.size(1)  # 실제 seq_len
        pe_slice = self.pe[:seq_len, :]
        pe_slice = pe_slice.unsqueeze(0) 
        # 최종 출력: 입력 + 위치 인코딩, 이후 드롭아웃
        x = x + pe_slice
        return self.dropout(x)