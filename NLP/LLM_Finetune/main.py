import torch
from torch.utils.data import DataLoader

from util import load_model_and_tokenizer
from dataset import load_qna_list, get_max_length, MyDataset
from train import train_model
from generate import generate_responses, interactive_generation

def main():
    # 디바이스 설정 및 랜덤 시드 고정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(123)
    
    # 모델과 토크나이저 로드
    model, tokenizer = load_model_and_tokenizer(device=device)
    
    # QnA 데이터 로드 및 최대 길이 계산
    qna_list = load_qna_list("./NLP/LLM_Finetune/data/qa_customdata.txt", tokenizer)
    max_length = get_max_length(qna_list)
    print(f"Max length: {max_length}")
    
    # 파인튜닝 전 응답 확인
    pre_questions = [qna['q'] for qna in qna_list]
    pre_questions.extend([
        "박윤수의 행운의 숫자는?",
        "박윤수가 좋아하는 디저트는 무엇인가요?",
        "박윤수의 취미 생활은?",
        "박윤수가 좋아하는 음악 장르는?",
        "박윤수가 좋아하는 동물은?"
    ])
    print("\nPre-finetuning responses:")
    generate_responses(model, tokenizer, pre_questions, device)
    
    # 데이터셋 및 DataLoader 준비
    dataset = MyDataset(qna_list, max_length)
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True, drop_last=False)
    
    # 모델 파인튜닝
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    print("\nStarting training...")
    train_model(model, train_loader, optimizer, device, epochs=10)
    
    # 파인튜닝 후 마지막 모델 상태 로드
    model.load_state_dict(torch.load("final_model.pth", map_location=device))
    
    # 파인튜닝 후 응답 확인
    post_questions = [qna['q'] for qna in qna_list]
    post_questions.extend([
        "박윤수의 행운의 숫자는?",
        "박윤수가 좋아하는 디저트는 무엇인가요?",
        "박윤수의 취미 생활은?",
        "박윤수가 좋아하는 음악 장르는?",
        "박윤수가 좋아하는 동물은?"
    ])
    print("\nPost-finetuning responses:")
    generate_responses(model, tokenizer, post_questions, device)
    
    # # 인터랙티브 응답 생성 (옵션)
    # interactive_generation(model, tokenizer, device)

if __name__ == "__main__":
    main()