import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from model import ConvVAE

IMAGE_DIR = r"..\training_images"
CHECKPOINT = "checkpoints/vae_cifar10.pt"

def main():
    torch.manual_seed(42)
    torch.set_num_threads(2)

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    images = []
    for name in sorted(os.listdir(IMAGE_DIR)):
        if name.lower().endswith((".jpg", ".jpeg", ".jfif", ".png", ".bmp", ".webp")):
            try:
                with Image.open(os.path.join(IMAGE_DIR, name)) as im:
                    images.append(transform(im.convert("RGB")))
                    print("Loaded:", name)
            except Exception as e:
                print("Skipped:", name, e)

    if len(images) < 2:
        raise SystemExit("ERROR: Need at least two local images.")

    loader = DataLoader(
        TensorDataset(torch.stack(images)),
        batch_size=min(4, len(images)),
        shuffle=True
    )

    model = ConvVAE(latent_dim=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)

    print("Training improved reconstruction model...")
    for epoch in range(30):
        model.train()
        total = 0.0

        for (clean,) in loader:
            sigma = torch.empty(1).uniform_(0.02, 0.12).item()
            noisy = (clean + torch.randn_like(clean) * sigma).clamp(0, 1)

            recon, mu, logvar = model(noisy)
            pixel_loss = F.mse_loss(recon, clean)
            kl_loss = -0.5 * torch.mean(
                1 + logvar - mu.pow(2) - logvar.exp()
            )
            loss = pixel_loss + 0.00001 * kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()

        print(f"Epoch {epoch+1}/30 | Loss: {total / len(loader):.5f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        "model_state_dict": model.cpu().state_dict(),
        "latent_dim": 128,
        "image_size": 64,
        "dataset": "Local images, noisy input and clean target"
    }, CHECKPOINT)

    print("SUCCESS: Improved model saved to", CHECKPOINT)

if __name__ == "__main__":
    main()
