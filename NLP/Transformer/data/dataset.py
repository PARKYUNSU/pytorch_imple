import os
import urllib.request
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast

# 토큰 정의 (필요에 따라 수정)
BOS = "</s>"
EOS = "</s>"
PAD = "<pad>"
MASK = "<unused0>"
Q_TKN = "<usr>"     # 질문 시작 토큰 (예시)
A_TKN = "<sys>"     # 답변 시작 토큰 (예시)
SENT = "<unused1>"  # 문장 구분 토큰 (예시)

# KoGPT2 토크나이저 로드
koGPT2_TOKENIZER = PreTrainedTokenizerFast.from_pretrained(
    "skt/kogpt2-base-v2",
    bos_token=BOS,
    eos_token=EOS,
    unk_token="<unk>",
    pad_token=PAD,
    mask_token=MASK,
)

class ChatbotDataset(Dataset):
    def __init__(self, chats, max_len=40):
        self._data = chats
        self.max_len = max_len
        self.q_token = Q_TKN
        self.a_token = A_TKN
        self.sent_token = SENT
        self.eos = EOS
        self.mask = MASK
        self.tokenizer = koGPT2_TOKENIZER

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        turn = self._data.iloc[idx]
        q = turn["Q"]
        q = re.sub(r"([?.!,])", r" ", q)

        a = turn["A"]
        a = re.sub(r"([?.!,])", r" ", a)

        # 질문과 답변 토큰화 및 특수 토큰 추가
        q_toked = self.tokenizer.tokenize(self.q_token + q + self.sent_token)
        q_len = len(q_toked)

        a_toked = self.tokenizer.tokenize(self.a_token + a + self.eos)
        a_len = len(a_toked)

        # 질문과 답변의 총 길이가 max_len을 초과하지 않도록 조절
        if q_len > self.max_len:
            a_len = self.max_len - q_len
            if a_len <= 0:
                q_toked = q_toked[-(int(self.max_len / 2)):]
                q_len = len(q_toked)
                a_len = self.max_seq_len - q_len
            a_toked = a_toked[:a_len]
            a_len = len(a_toked)

        if q_len + a_len > self.max_len:
            a_len = self.max_len - q_len
            if a_len <= 0:
                q_toked = q_toked[-(int(self.max_len / 2)):]
                q_len = len(q_toked)
                a_len = self.max_len - q_len
            a_toked = a_toked[:a_len]
            a_len = len(a_toked)

        # 답변 부분은 모델이 예측해야 하므로 라벨 생성 (질문 부분은 MASK 토큰으로 채움)
        labels = [self.mask] * q_len + a_toked[1:]
        mask = [0] * q_len + [1] * a_len + [0] * (self.max_len - q_len - a_len)

        # 토큰들을 ID로 변환하고, 최대 길이에 맞게 패딩 수행
        labels_ids = self.tokenizer.convert_tokens_to_ids(labels)
        while len(labels_ids) < self.max_len:
            labels_ids += [self.tokenizer.pad_token_id]

        token_ids = self.tokenizer.convert_tokens_to_ids(q_toked + a_toked)
        while len(token_ids) < self.max_len:
            token_ids += [self.tokenizer.pad_token_id]

        return (token_ids, np.array(mask), labels_ids)

def collate_batch(batch):
    data = np.array([item[0] for item in batch])
    mask = np.array([item[1] for item in batch])
    label = np.array([item[2] for item in batch])
    return torch.LongTensor(data), torch.LongTensor(mask), torch.LongTensor(label)

def download_data(url="https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv",
                  filename="ChatBotData.csv"):
    if not os.path.exists(filename):
        print(f"'{filename}' 파일이 존재하지 않습니다. 다운로드를 시작합니다.")
        urllib.request.urlretrieve(url, filename=filename)
        print("다운로드 완료!")
    else:
        print(f"'{filename}' 파일이 이미 존재합니다.")
    return filename

# --- create_mask 함수 추가 ---
def create_mask(src, tgt, pad_idx):
    # src: [batch, src_seq_len], tgt: [batch, tgt_seq_len]
    # src_mask: padding mask for encoder inputs
    src_mask = (src != pad_idx).unsqueeze(1)  # (batch, 1, src_seq_len)
    
    # tgt_mask: padding mask for decoder inputs
    tgt_mask = (tgt != pad_idx).unsqueeze(1)  # (batch, 1, tgt_seq_len)
    
    # 생성 마스크: decoder의 미래 정보를 보지 않도록 함 (upper triangular mask)
    tgt_seq_len = tgt.size(1)
    subsequent_mask = torch.triu(torch.ones((tgt_seq_len, tgt_seq_len), device=tgt.device), diagonal=1).bool()
    combined_tgt_mask = tgt_mask & ~subsequent_mask

    memory_mask = None  # 보통은 사용하지 않거나 별도로 구현
    return src_mask, combined_tgt_mask, memory_mask

if __name__ == "__main__":
    # 데이터 파일 다운로드 (필요한 경우)
    filename = download_data()
    # CSV 파일 읽기
    df = pd.read_csv(filename)
    # 테스트용으로 상위 300개 샘플만 사용
    df = df[:300]
    print("데이터 미리보기:")
    print(df.head())