import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

def train_model(model, train_loader, optimizer, device, epochs=10, model_save_path="final_model.pth", loss_plot_path="loss_plot.png"):
    tokens_seen, global_step = 0, -1
    losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for input_batch, target_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            optimizer.zero_grad()
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            logits = model(input_batch).logits
            loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

            tokens_seen += input_batch.numel()
            global_step += 1
            tqdm.write(f"{global_step} Tokens seen: {tokens_seen}")

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"Epoch: {epoch}, Loss: {avg_loss}")

    # 학습 완료 후 최종 모델 저장
    torch.save(model.state_dict(), model_save_path)
    print(f"Final model saved to {model_save_path}")

    # 손실 플롯 생성 및 저장
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.savefig(loss_plot_path)
    plt.show()
    print(f"Loss plot saved to {loss_plot_path}")
    
    return losses