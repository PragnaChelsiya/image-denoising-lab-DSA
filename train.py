import argparse
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from model import ConvVAE

def loss_function(recon, x, mu, logvar):
    # Mean reconstruction loss plus KL divergence; both terms train the VAE.
    recon_loss = nn.functional.mse_loss(recon, x, reduction="sum") / x.size(0)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + (0.001 * kl)

def main():
    parser = argparse.ArgumentParser(description="Train the convolutional VAE on CIFAR-10.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=20000, help="training subset size; use 0 for full dataset")
    parser.add_argument("--latent-dim", type=int, default=128)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
    if args.limit and args.limit < len(dataset):
        dataset = Subset(dataset, range(args.limit))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = ConvVAE(args.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        for images, _ in loader:
            images = images.to(device)
            recon, mu, logvar = model(images)
            loss = loss_function(recon, images, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"Epoch {epoch+1}/{args.epochs} - average loss: {total / len(loader):.4f}")
    os.makedirs("checkpoints", exist_ok=True)
    path = "checkpoints/vae_cifar10.pt"
    torch.save({"model_state_dict": model.cpu().state_dict(), "latent_dim": args.latent_dim,
                "image_size": 64, "dataset": "CIFAR-10"}, path)
    print(f"Saved trained model to {path}")

if __name__ == "__main__":
    main()
