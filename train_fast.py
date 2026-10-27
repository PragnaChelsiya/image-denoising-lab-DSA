import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from model import ConvVAE

torch.manual_seed(42)
torch.set_num_threads(2)

print("Creating local synthetic training images...")
images = F.interpolate(
    torch.rand(64, 3, 8, 8),
    size=(64, 64),
    mode="bilinear",
    align_corners=False
).clamp(0, 1)

loader = DataLoader(TensorDataset(images), batch_size=16, shuffle=True)
model = ConvVAE(latent_dim=128)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()
for epoch in range(1):
    total_loss = 0.0
    for (batch,) in loader:
        reconstruction, mu, logvar = model(batch)
        recon_loss = F.mse_loss(reconstruction, batch)
        kl_loss = -0.5 * torch.mean(
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        loss = recon_loss + 0.001 * kl_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch 1/1 complete. Loss: {total_loss / len(loader):.5f}")

os.makedirs("checkpoints", exist_ok=True)
torch.save({
    "model_state_dict": model.state_dict(),
    "latent_dim": 128,
    "image_size": 64,
    "dataset": "Synthetic demo patterns"
}, "checkpoints/vae_cifar10.pt")

print("SUCCESS: Model saved to checkpoints/vae_cifar10.pt")
