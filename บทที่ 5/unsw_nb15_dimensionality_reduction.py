
# -*- coding: utf-8 -*-
# Workshop: Dimensionality Reduction on UNSW-NB15 Dataset
# Includes PCA, t-SNE, UMAP for cybersecurity anomaly visualization
# Author: ChatGPT Cybersecurity Research Assistant

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
import seaborn as sns

# 1. โหลดข้อมูล UNSW-NB15 (แนะนำให้อยู่ในรูปแบบ .csv)
# ตัวอย่างจำลอง: โปรดแทน path ด้วยตำแหน่งจริงของไฟล์
df = pd.read_csv('unsw_nb15_cleaned.csv')  # ต้องมีคอลัมน์ 'label'

# 2. เตรียมข้อมูล
X = df.drop('label', axis=1)
y = df['label']  # 0 = normal, 1 = anomaly

# 3. Standardize ข้อมูล
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
pca_df["label"] = y

# 5. t-SNE
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)
tsne_df = pd.DataFrame(X_tsne, columns=["Dim1", "Dim2"])
tsne_df["label"] = y

# 6. UMAP
reducer = umap.UMAP(n_neighbors=15, n_components=2, random_state=42)
X_umap = reducer.fit_transform(X_scaled)
umap_df = pd.DataFrame(X_umap, columns=["UMAP1", "UMAP2"])
umap_df["label"] = y

# 7. แสดงผล
def plot_result(df, x, y, title):
    plt.figure(figsize=(7,5))
    sns.scatterplot(data=df, x=x, y=y, hue="label", palette="Set2", alpha=0.8)
    plt.title(title)
    plt.tight_layout()
    plt.show()

plot_result(pca_df, "PC1", "PC2", "PCA on UNSW-NB15")
plot_result(tsne_df, "Dim1", "Dim2", "t-SNE on UNSW-NB15")
plot_result(umap_df, "UMAP1", "UMAP2", "UMAP on UNSW-NB15")
