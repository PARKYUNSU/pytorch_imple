import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler

from util import load_model_and_tokenizer
from dataset import load_qna_list, get_max_length, MyDataset
from train import train_model
from generate import generate_responses, interactive_generation

def main_worker(local_rank, world_size):
    # 각 프로세스가 사용할 GPU 설정
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    # 분산 프로세스 그룹 초기화 (NCCL 백엔드 사용)
    dist.init_process_group(
        backend='nccl',
        init_method='env://',  # torchrun 또는 환경변수를 이용하여 초기화
        world_size=world_size,
        rank=local_rank
    )
    
    # 모델과 토크나이저 로드 후 DDP 래핑
    model, tokenizer = load_model_and_tokenizer(device=device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    
    torch.manual_seed(123)
    
    # QnA 데이터 로드 및 최대 길이 계산 (모든 프로세스 동일한 데이터 사용)
    qna_list = load_qna_list("./NLP/LLM_Finetune/data/qa_customdata.txt", tokenizer)
    max_length = get_max_length(qna_list)
    if local_rank == 0:
        print(f"Max length: {max_length}")
    
    # 파인튜닝 전 응답 확인 (rank 0에서만)
    if local_rank == 0:
        pre_questions = [qna['q'] for qna in qna_list]
        pre_questions.extend([
            "너에 대해서 설명해봐.",
            "이처럼 인간처럼 생각하고 행동하는 AI 모델은 "
        ])
        print("\nPre-finetuning responses:")
        generate_responses(model, tokenizer, pre_questions, device)
    
    # 데이터셋 및 DataLoader 준비 (DistributedSampler 사용)
    dataset = MyDataset(qna_list, max_length)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
    train_loader = DataLoader(dataset, batch_size=2, sampler=sampler, drop_last=False)
    
    # 모델 파인튜닝
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    if local_rank == 0:
        print("\nStarting training...")
    train_model(model, train_loader, optimizer, device, epochs=10)
    
    # 학습 완료 후, rank 0에서만 최종 모델 저장 및 응답 생성
    if local_rank == 0:
        state_dict = model.module.state_dict()  # DDP인 경우 내부 모델 저장
        torch.save(state_dict, "final_model.pth")
        print("Final model saved to final_model.pth")
        
        # 저장한 모델 상태 로드 (테스트를 위해)
        state_dict = torch.load("final_model.pth", map_location=device)
        model.module.load_state_dict(state_dict)
        
        post_questions = [qna['q'] for qna in qna_list]
        post_questions.extend([
            "러시아에 대해서 얘기해봐",
            "1+1은 뭐야?",
            "인간은 뭐라고 생각해?",
            "수면이 부족하면 뭘 해야해?",
            "인공지능의 장점은"
        ])
        print("\nPost-finetuning responses:")
        generate_responses(model, tokenizer, post_questions, device)
        interactive_generation(model, tokenizer, device)
    
    # 분산 프로세스 그룹 종료
    dist.destroy_process_group()

def main():
    world_size = torch.cuda.device_count()
    mp.spawn(main_worker, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()