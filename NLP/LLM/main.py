import argparse
import torch
import tiktoken
from data_processing import clean_text
from dataset import MyDataset
from train import train_model
from generate import generate
from util import plot_losses

parser = argparse.ArgumentParser(description="Train and generate text with different GPT models.")
parser.add_argument("--model", type=str, default="base", choices=["base", "pre-norm"],
                    help="Select model type to use.")
args = parser.parse_args()

if args.model == "pre-norm":
    from model.pre_norm import GPTModel, CONTEXT_LENGTH
else:
    from model.base import GPTModel, CONTEXT_LENGTH

# 텍스트 파일 정제
filename = "./NLP/LLM/data/02 Harry Potter and the Chamber of Secrets.txt"
cleaned_filename = clean_text(filename)

# 정제된 텍스트 불러오기
with open(cleaned_filename, 'r', encoding='utf-8-sig') as file:
    txt = file.read()

# 토크나이저 설정 (tiktoken 사용)
tokenizer = tiktoken.get_encoding("gpt2")

# 데이터셋 생성
dataset = MyDataset(txt, tokenizer, max_length=32, stride=4)

# 모델 초기화 및 학습 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPTModel()

epochs = 100
batch_size = 128
lr = 0.0004
losses = train_model(model, dataset, epochs, batch_size, lr, device)
plot_losses(losses, filename="training_loss.png")

# 이미 학습된 모델 불러오기
model.load_state_dict(torch.load("model_final.pth", map_location=device))
model.eval()

# 간단한 텍스트 생성 예시
start_context = input("Start context: ")
idx = tokenizer.encode(start_context)
idx = torch.tensor(idx).unsqueeze(0).to(device)
generated_ids = generate(model, idx, max_new_tokens=50, context_size=CONTEXT_LENGTH, top_k=50, temperature=0.5)
output_text = tokenizer.decode(generated_ids.squeeze(0).tolist()).replace("\n", " ")
print("Generated text:", output_text)