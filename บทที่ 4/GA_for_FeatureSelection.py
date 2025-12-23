import random
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# === สร้างข้อมูลจำลอง (แทน Traffic เครือข่าย) ===
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

n_features = X.shape[1]
population_size = 10
generations = 30
mutation_rate = 0.1

# === ฟังก์ชันฟิตเนส: Accuracy ของ Logistic Regression ===
def fitness(chromosome):
    selected_indices = [i for i, gene in enumerate(chromosome) if gene == 1]
    if len(selected_indices) == 0:
        return 0  # ไม่มีฟีเจอร์ใดเลย
    model = LogisticRegression()
    model.fit(X_train[:, selected_indices], y_train)
    y_pred = model.predict(X_val[:, selected_indices])
    return accuracy_score(y_val, y_pred)

# === สร้างโครโมโซมเริ่มต้น ===
def init_population():
    return [random.choices([0, 1], k=n_features) for _ in range(population_size)]

# === การเลือกแบบ Tournament ===
def tournament_selection(pop, fitnesses, k=3):
    selected = random.sample(list(zip(pop, fitnesses)), k)
    return max(selected, key=lambda x: x[1])[0]

# === Crossover: Single-point ===
def crossover(p1, p2):
    point = random.randint(1, len(p1) - 1)
    return p1[:point] + p2[point:], p2[:point] + p1[point:]

# === Mutation: Flip บิตแต่ละตำแหน่งแบบสุ่ม ===
def mutate(chromosome):
    return [1 - gene if random.random() < mutation_rate else gene for gene in chromosome]

# === Decode ===
def decode_chromosome(chromosome, feature_names):
    return [name for gene, name in zip(chromosome, feature_names) if gene == 1]

# === อัลกอริทึม GA หลัก ===
population = init_population()
print("Init Population\n")
print(population)
#print("\n")
best_chromosome = None
best_fitness = 0

for gen in range(generations):
    fitnesses = [fitness(chromo) for chromo in population]

    # บันทึกโครโมโซมที่ดีที่สุดในรุ่นนี้
    max_fit = max(fitnesses)
    if max_fit > best_fitness:
        best_fitness = max_fit
        best_chromosome = population[fitnesses.index(max_fit)]

    new_population = []
    while len(new_population) < population_size:
        p1 = tournament_selection(population, fitnesses)
        p2 = tournament_selection(population, fitnesses)
        c1, c2 = crossover(p1, p2)
        new_population.append(mutate(c1))
        new_population.append(mutate(c2))
    population = new_population[:population_size]
    print(f"Generation {gen+1}: Best Accuracy = {best_fitness:.4f}")

# === ผลลัพธ์สุดท้าย ===
selected_features = [i for i, gene in enumerate(best_chromosome) if gene == 1]
print("\n✅ Best Chromosome:", best_chromosome)
print("✅ Selected Features:", selected_features)
print("✅ Best Accuracy:", best_fitness)

feature_names = ["FlowDuration", "PacketSize", "Protocol", "Port", "BytesPerSecond"]
#chromosome = [1, 0, 1, 1, 0]
chromosome = best_chromosome

selected_features = decode_chromosome(chromosome, feature_names)
print("✅ Selected Feature:", selected_features)
