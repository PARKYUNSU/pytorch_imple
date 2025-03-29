import torch
from torch.utils.data import Dataset

EOT = 128001  # End Of Text token id

def load_qna_list(file_path, tokenizer):
    """
    파일에서 QnA 데이터를 읽어, 질문과 답변을 구분한 후 토크나이즈한 결과를 리스트로 반환합니다.
    각 줄은 '질문|답변' 형식입니다.
    """
    qna_list = []
    with open(file_path, "r") as file:
        for line in file:
            qna = line.strip().split('|')
            if len(qna) < 2:
                continue
            input_str = qna[0] + " " + qna[1]
            item = {
                'q': qna[0],
                'input': input_str,
                'q_ids': tokenizer.encode(qna[0]),
                'input_ids': tokenizer.encode(input_str)
            }
            qna_list.append(item)
    return qna_list

def get_max_length(qna_list):
    """
    qna_list 내 각 항목의 input_ids 길이 중 최댓값을 반환합니다.
    """
    return max(len(item['input_ids']) for item in qna_list)

class MyDataset(Dataset):
    """
    파인튜닝을 위한 Dataset 클래스.
    입력 시퀀스와 타깃 시퀀스를 생성하며, 질문 부분은 CrossEntropy 계산 시 무시(-100)하도록 마스킹합니다.
    """
    def __init__(self, qna_list, max_length):
        self.input_ids = []
        self.target_ids = []
        for qa in qna_list:
            token_ids = qa['input_ids']
            input_chunk = token_ids.copy()
            target_chunk = token_ids[1:].copy()  # 오른쪽 쉬프트
            # max_length까지 패딩
            input_chunk += [EOT] * (max_length - len(input_chunk))
            target_chunk += [EOT] * (max_length - len(target_chunk))
            # 질문 부분은 loss 계산 시 무시(-100) 처리 (질문 토큰 수 -1)
            len_ignore = len(qa['q_ids']) - 1
            target_chunk[:len_ignore] = [-100] * len_ignore

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]