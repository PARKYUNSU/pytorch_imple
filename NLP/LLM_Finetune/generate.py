import torch

def generate_responses(model, tokenizer, questions, device, max_new_tokens=32):
    """
    주어진 질문 리스트에 대해 모델의 응답을 생성하고 출력합니다.
    DataParallel 래핑이 되어 있다면 내부 모델(module)의 generate 메서드를 사용합니다.
    """
    responses = []
    # DataParallel 여부 확인
    model_to_use = model.module if hasattr(model, "module") else model

    for q in questions:
        input_ids = tokenizer(q, padding=True, return_tensors="pt")["input_ids"].to(device)
        model_to_use.eval()
        with torch.no_grad():
            output = model_to_use.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                attention_mask=(input_ids != 0).long(),
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )
        response_text = tokenizer.decode(output[0], skip_special_tokens=True)
        responses.append(response_text)
        print(f"Q: {response_text}")
    return responses

def interactive_generation(model, tokenizer, device, max_new_tokens=32):
    """
    사용자 입력을 받아 실시간으로 모델의 응답을 생성합니다.
    DataParallel 래핑이 되어 있다면 내부 모델(module)의 generate 메서드를 사용합니다.
    """
    user_input = input("Enter a prompt: ")
    input_ids = tokenizer(user_input, padding=True, return_tensors="pt")["input_ids"].to(device)
    model_to_use = model.module if hasattr(model, "module") else model
    model_to_use.eval()
    with torch.no_grad():
        output = model_to_use.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            attention_mask=(input_ids != 0).long(),
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Response: {response}")
    return response