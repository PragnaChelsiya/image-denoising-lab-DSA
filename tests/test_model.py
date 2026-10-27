import torch
from model import ConvVAE

def test_encoder_decoder_shapes_and_latent():
    model = ConvVAE(latent_dim=64).eval()
    x = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        mu, logvar = model.encode(x)
        output = model.decode(mu)
    assert mu.shape == (2, 64)
    assert logvar.shape == (2, 64)
    assert output.shape == x.shape
    assert torch.isfinite(output).all()

def test_forward_pass():
    model = ConvVAE(latent_dim=32).eval()
    x = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        reconstruction, mu, logvar = model(x, sample=False)
    assert reconstruction.shape == x.shape
    assert mu.shape == (1, 32)
    assert logvar.shape == (1, 32)
