import torch
from torch.utils.data import DataLoader
from model import GPTModel, CONTEXT_LENGTH

def train_model(model, dataset, epochs, batch_size, lr, device):
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    tokens_seen, global_step = 0, -1
    losses = []

    model.to(device)
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            logits = model(input_batch)
            loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1
            if global_step % 1000 == 0:
                print(f"Tokens seen: {tokens_seen}")
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"Epoch: {epoch + 1}, Loss: {avg_loss}")
        torch.save(model.state_dict(), f"model_{str(epoch + 1).zfill(3)}.pth")
    return losses