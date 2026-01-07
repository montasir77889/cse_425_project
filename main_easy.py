import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from src.feature_extraction import build_dataset
from src.vae import VAE, vae_loss
from src.clustering import cluster_and_save
from src.baseline_pca import evaluate_and_save

# folders
os.makedirs("results/figures", exist_ok=True)

# Step 1: Load & normalize data
X = build_dataset(r"E:\425\Datasets")
X = X.reshape(X.shape[0], -1)
X = (X - X.mean()) / X.std()

X_tensor = torch.tensor(X, dtype=torch.float32)
loader = DataLoader(TensorDataset(X_tensor), batch_size=16, shuffle=True)

# Step 2: Train VAE
vae = VAE(input_dim=X.shape[1])
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

losses = []
for epoch in range(70):
    epoch_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        recon, mu, logvar = vae(batch[0])
        loss = vae_loss(recon, batch[0], mu, logvar)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    losses.append(epoch_loss)
    print(f"Epoch {epoch+1}, Loss: {epoch_loss:.2f}")

# Save loss curve
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("VAE Training Loss")
plt.savefig("results/figures/vae_training_loss.png", dpi=300)
plt.close()

# Step 3: Extract latent space
vae.eval()
with torch.no_grad():
    mu, _ = vae.encode(X_tensor)
z = mu.numpy()

# Step 4: Clustering & visualization
cluster_and_save(z, "results/figures/vae_tsne_clusters.png")

# Step 5: Baseline comparison
evaluate_and_save(X, z)
