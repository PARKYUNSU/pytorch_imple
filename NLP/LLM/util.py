import matplotlib.pyplot as plt

def plot_losses(losses, filename="training_loss.png"):
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.savefig(filename)
    plt.close()
    print(f"Loss plot saved as {filename}")