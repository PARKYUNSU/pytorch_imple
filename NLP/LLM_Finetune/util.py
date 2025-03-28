import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

EOT = 128001  # End Of Text token id
MODEL_NAME = "kakaocorp/kanana-nano-2.1b-base"

def load_model_and_tokenizer(model_name=MODEL_NAME, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    모델과 토크나이저를 로드합니다.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token  # pad_token을 eos_token으로 설정
    return model, tokenizer