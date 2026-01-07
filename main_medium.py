import sys, os
sys.path.insert(0, os.path.abspath("."))

import torch
import numpy as np
import pandas as pd

from src.audio_features import load_audio_mels
from src.cnn_vae import CNNVAE
from src.text_features import extract_lyrics_embeddings
from src.fusion import fuse_features
from src.clustering_medium import cluster_features
from src.visualize import plot_tsne

# ---------- GPU ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------- Load Audio ----------
audio = load_audio_mels(r"E:\425\Datasets")
audio = audio[:, None, :, :]
audio = torch.tensor(audio, dtype=torch.float32).to(device)

# ---------- CNN-VAE ----------
model = CNNVAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

batch_size = 32
model.train()

for epoch in range(50):
    perm = torch.randperm(audio.size(0))
    total_loss = 0

    for i in range(0, audio.size(0), batch_size):
        idx = perm[i:i+batch_size]
        batch = audio[idx]

        optimizer.zero_grad()
        recon, mu, logvar = model(batch)

        recon_loss = ((recon - batch) ** 2).mean()
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + 0.001 * kl_loss

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

# ---------- Latent Features ----------
model.eval()
with torch.no_grad():
    _, mu, _ = model(audio)

audio_latent = mu.cpu().numpy()

# ---------- Lyrics ----------
lyrics = extract_lyrics_embeddings(r"E:\425\Datasets\updated_metadata.csv")

# ---------- Align ----------
N = min(len(audio_latent), len(lyrics))
audio_latent = audio_latent[:N]
lyrics = lyrics[:N]

# ---------- Fusion ----------
fused = fuse_features(audio_latent, lyrics)

# ---------- Remove NaNs (FINAL SAFETY) ----------
mask = ~np.isnan(fused).any(axis=1)
fused = fused[mask]

# ---------- Clustering ----------
labels, sil, ch = cluster_features(fused)

# ---------- Save Outputs ----------
os.makedirs("results/embeddings", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/tables", exist_ok=True)

np.save("results/embeddings/audio_latent.npy", audio_latent)
np.save("results/embeddings/lyrics_embeddings.npy", lyrics)
np.save("results/embeddings/fused_features.npy", fused)

pd.DataFrame({
    "Silhouette": [sil],
    "Calinski_Harabasz": [ch]
}).to_csv("results/tables/medium_metrics.csv", index=False)

plot_tsne(fused, labels, "results/figures/medium_tsne.png")

print("✅ MEDIUM TASK COMPLETED SUCCESSFULLY")
