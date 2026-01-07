import sys, os
sys.path.insert(0, os.path.abspath("."))

import torch
import numpy as np
import pandas as pd

from src.audio_features import load_audio_mels
from src.beta_vae import BetaVAE
from src.text_features import extract_lyrics_embeddings
from src.genre_features import extract_genre_features
from src.fusion import fuse_features
from src.clustering_hard import cluster
from src.evaluation_hard import evaluate
from src.visualize_hard import plot_latent

# ---------- GPU ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- AUDIO ----------
audio = load_audio_mels(r"E:\425\Datasets")
audio = torch.tensor(audio[:,None,:,:], dtype=torch.float32).to(device)

# ---------- BETA-VAE ----------
model = BetaVAE(beta=4.0).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(50):
    opt.zero_grad()
    recon, mu, logvar = model(audio)
    loss = model.loss(recon, audio, mu, logvar)
    loss.backward()
    opt.step()
    print(f"Epoch {epoch+1} Loss {loss.item():.4f}")

z = mu.detach().cpu().numpy()

# ---------- LYRICS ----------
lyrics = extract_lyrics_embeddings(r"E:\425\Datasets\updated_metadata.csv")

# ---------- GENRE ----------
genre_vec, genre_names = extract_genre_features(r"E:\425\Datasets\updated_metadata.csv")

# ---------- ALIGN ----------
N = min(len(z), len(lyrics), len(genre_vec))
z, lyrics, genre_vec = z[:N], lyrics[:N], genre_vec[:N]

# ---------- FUSION ----------
fused = np.concatenate([z, lyrics, genre_vec], axis=1)

# ---------- CLUSTER ----------
labels, sil = cluster(fused)

# ---------- EVALUATION ----------
df = pd.read_csv(r"E:\425\Datasets\updated_metadata.csv")
true_genre = df["genre"][:N].factorize()[0]
metrics = evaluate(true_genre, labels)

# ---------- SAVE ----------
os.makedirs("results/embeddings", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/tables", exist_ok=True)

np.save("results/embeddings/beta_vae_latent.npy", z)
np.save("results/embeddings/fused_hard.npy", fused)

pd.DataFrame([{**metrics, "Silhouette": sil}]).to_csv(
    "results/tables/hard_metrics.csv", index=False
)

plot_latent(z, labels, "results/figures/hard_latent_tsne.png")

print("✅ HARD TASK COMPLETED")
