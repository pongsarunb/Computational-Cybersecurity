import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# สร้างข้อมูลจำลอง
X, y = make_classification(n_samples=1000, n_features=10, n_informative=7, random_state=42)
X = StandardScaler().fit_transform(X)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# พารามิเตอร์ Binary PSO
n_particles = 15
n_features = X.shape[1]
max_iter = 30
w = 0.5
c1 = 1.5
c2 = 1.5

# ฟังก์ชัน sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ฟังก์ชันฟิตเนส
def fitness(binary_vector):
    selected = [i for i, bit in enumerate(binary_vector) if bit == 1]
    if not selected:
        return 0
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train[:, selected], y_train)
    y_pred = model.predict(X_val[:, selected])
    return accuracy_score(y_val, y_pred)

# Initialization
positions = np.random.randint(0, 2, size=(n_particles, n_features))
velocities = np.random.uniform(-1, 1, size=(n_particles, n_features))
p_best = positions.copy()
p_best_scores = np.array([fitness(p) for p in p_best])
g_best = p_best[np.argmax(p_best_scores)]
g_best_score = np.max(p_best_scores)
history = [g_best_score]

# === Decode ===
def decode_chromosome(chromosome, feature_names):
    return [name for gene, name in zip(chromosome, feature_names) if gene == 1]

# PSO Loop
for t in range(max_iter):
    for i in range(n_particles):
        r1, r2 = np.random.rand(n_features), np.random.rand(n_features)
        velocities[i] = (
            w * velocities[i]
            + c1 * r1 * (p_best[i] - positions[i])
            + c2 * r2 * (g_best - positions[i])
        )
        prob = sigmoid(velocities[i])
        positions[i] = [1 if np.random.rand() < p else 0 for p in prob]

        score = fitness(positions[i])
        if score > p_best_scores[i]:
            p_best[i] = positions[i]
            p_best_scores[i] = score

    if np.max(p_best_scores) > g_best_score:
        g_best = p_best[np.argmax(p_best_scores)]
        g_best_score = np.max(p_best_scores)

    history.append(g_best_score)

# แสดงผลลัพธ์
print("\n✅ ฟีเจอร์ที่ถูกเลือก:", g_best)
print("✅ Best Accuracy: {:.4f}".format(g_best_score))

feature_names = ["FlowDuration", "PacketSize", "Protocol", "Port", "BytesPerSecond"]
chromosome = g_best

selected_features = decode_chromosome(chromosome, feature_names)
print("✅ Selected Feature:", selected_features)

# กราฟความแม่นยำ
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(history)+1), history, marker='o')
plt.title("Best Accuracy over Generations (Binary PSO)")
plt.xlabel("Iteration")
plt.ylabel("Best Accuracy")
plt.grid(True)
plt.show()
