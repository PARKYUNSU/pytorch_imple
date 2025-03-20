# inference.py
import torch
from torch import nn
import torch.nn.functional as F
import argparse

from model.transformer import Transformer
from data.dataset import koGPT2_TOKENIZER  # koGPT2_TOKENIZER를 data/dataset.py에서 가져옵니다.

# 사전 정의된 특수 토큰 (데이터셋과 일치)
Q_TKN = "<usr>"  # 질문 시작 토큰
A_TKN = "<sys>"  # 답변 시작 토큰
SENT  = "<unused1>"  # 문장 구분 토큰
EOS   = koGPT2_TOKENIZER.eos_token  # EOS 토큰 (예: "</s>")

def load_model(checkpoint_path, args, device):
    model = Transformer(
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        vocab_size=koGPT2_TOKENIZER.vocab_size,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout
    ).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

def greedy_decode(model, encoder_output, decoder_input_ids, max_len, pad_token_id, eos_token_id, device, num_heads):
    """
    Greedy decoding: 주어진 decoder_input_ids에서 시작하여, EOS 토큰이 생성될 때까지 토큰을 한 개씩 생성.
    encoder_output은 미리 계산된 encoder의 결과입니다.
    """
    generated = decoder_input_ids  # [1, seq_len_init]
    for _ in range(max_len - decoder_input_ids.size(1)):
        # 현재 decoder 입력에 대한 mask 생성 (여기서는 간단히 causal mask 생성)
        current_seq_len = generated.size(1)
        # causal mask: [current_seq_len, current_seq_len]
        tgt_mask = torch.triu(torch.ones((current_seq_len, current_seq_len), device=device), diagonal=1).bool()
        # 모델이 기대하는 shape: [batch, num_heads, seq_len, seq_len]
        tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(1).expand(1, num_heads, current_seq_len, current_seq_len)
        
        # decoder 출력: [1, current_seq_len, vocab_size]
        outputs = model.decoder(generated, encoder_output, tgt_mask=tgt_mask, memory_mask=None)
        # 마지막 토큰에 해당하는 logits: [vocab_size]
        next_token_logits = outputs[0, -1, :]
        # Greedy 선택
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0).unsqueeze(0)  # [1,1]
        generated = torch.cat([generated, next_token], dim=1)
        if next_token.item() == eos_token_id:
            break
    return generated

def inference(model, tokenizer, device, args):
    num_heads = args.num_heads
    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id

    print("챗봇과 대화를 시작합니다. 종료하려면 'quit' 또는 'exit'를 입력하세요.\n")
    while True:
        question = input("User: ")
        if question.lower() in ["quit", "exit"]:
            break

        # 입력 전처리: 질문 토큰화, 특수 토큰 추가 등
        # 예: "<usr>" + 질문 + "<unused1>"
        input_text = Q_TKN + question + SENT
        tokens = tokenizer.tokenize(input_text)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # 입력 길이가 max_seq_len을 넘으면 자르고, 부족하면 패딩
        if len(input_ids) > args.max_seq_len:
            input_ids = input_ids[:args.max_seq_len]
        else:
            input_ids += [pad_token_id] * (args.max_seq_len - len(input_ids))
        encoder_input = torch.LongTensor(input_ids).unsqueeze(0).to(device)  # [1, max_seq_len]

        # encoder 실행: encoder_output: [1, max_seq_len, d_model]
        encoder_output = model.encoder(encoder_input)

        # 디코더 입력 초기화: 일반적으로 시스템 응답 시작 토큰 "<sys>" 사용
        dec_input = A_TKN  # 시작 토큰
        dec_tokens = tokenizer.tokenize(dec_input)
        dec_input_ids = tokenizer.convert_tokens_to_ids(dec_tokens)
        dec_input_ids = torch.LongTensor(dec_input_ids).unsqueeze(0).to(device)  # [1, 1]

        # greedy decoding 실행
        generated_ids = greedy_decode(model, encoder_output, dec_input_ids,
                                      max_len=args.max_seq_len,
                                      pad_token_id=pad_token_id,
                                      eos_token_id=eos_token_id,
                                      device=device,
                                      num_heads=num_heads)
        # 디코딩: 생성된 토큰 시퀀스를 문자열로 변환 (특수 토큰 제거)
        generated_text = tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)
        print("Bot:", generated_text)
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='학습된 모델 checkpoint 경로')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=32)
    parser.add_argument('--d_ff', type=int, default=256)
    parser.add_argument('--max_seq_len', type=int, default=50)
    parser.add_argument('--dropout', type=float, default=0.1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, args, device)
    inference(model, koGPT2_TOKENIZER, device, args)