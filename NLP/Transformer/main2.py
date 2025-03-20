import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader

from data.dataset import ChatbotDataset, collate_batch, download_data, koGPT2_TOKENIZER
from train import train_one_epoch, validate_one_epoch
from model.transformer import Transformer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=32)
    parser.add_argument('--d_ff', type=int, default=256)
    parser.add_argument('--max_seq_len', type=int, default=50)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--total_samples', type=int, default=1000)
    parser.add_argument('--val_samples', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--save_fig', action='store_true', help='If set, saves the loss plot as a PNG file')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    filename = download_data()
    df = pd.read_csv(filename)

    train_df = df[:args.total_samples]
    val_df = df[:args.val_samples]

    train_dataset = ChatbotDataset(train_df, max_len=args.max_seq_len)
    val_dataset = ChatbotDataset(val_df, max_len=args.max_seq_len)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch, num_workers=0)

    model = Transformer(
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        vocab_size=koGPT2_TOKENIZER.vocab_size,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_losses = []
    val_losses = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_dataloader, criterion, optimizer, device, args.num_heads)
        val_loss = validate_one_epoch(model, val_dataloader, criterion, device, args.num_heads)

        print(f"Epoch [{epoch}/{args.epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
    torch.save(model.state_dict(), "final_checkpoint.pth")
    print("Final checkpoint saved as 'final_checkpoint.pth'")

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, args.epochs + 1), train_losses, marker='o', label='Train Loss')
    plt.plot(range(1, args.epochs + 1), val_losses, marker='x', label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    if args.save_fig:
        plt.savefig('loss_plot.png')
        print("Loss plot saved as 'loss_plot.png'")
    else:
        plt.show()

if __name__ == '__main__':
    main()