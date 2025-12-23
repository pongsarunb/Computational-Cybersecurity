import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
import seaborn as sns

# สมมุติว่าคุณมีไฟล์ unsw_nb15_clean.csv (เฉพาะฟีเจอร์เชิงตัวเลข)
df = pd.read_csv("unsw_nb15_cleaned.csv")  # ลบ column ที่ไม่เกี่ยวข้องไว้ก่อน
X = df.drop(columns="label")  # ห้ามมี IP, proto, service ฯลฯ
y = df["label"]

# 1. Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 3. t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)

# 4. UMAP
umap_model = umap.UMAP(n_components=2, random_state=42)
X_umap = umap_model.fit_transform(X_scaled)

# สร้างกราฟ
def plot_2d(X_proj, title):
    df_plot = pd.DataFrame(X_proj, columns=["Dim1", "Dim2"])
    df_plot["label"] = y
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_plot, x="Dim1", y="Dim2", hue="label", palette="Set2", s=40, alpha=0.6)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_2d(X_pca, "PCA Projection")
plot_2d(X_tsne, "t-SNE Projection")
plot_2d(X_umap, "UMAP Projection")
