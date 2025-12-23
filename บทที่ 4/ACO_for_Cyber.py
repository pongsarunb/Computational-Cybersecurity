import numpy as np
import random
import matplotlib.pyplot as plt

# === STEP 1: สร้างข้อมูลจำลองเครือข่าย ===
n_nodes = 10
np.random.seed(42)
random.seed(42)

# Matrix เวลาเดินทางระหว่างอุปกรณ์ (Time Matrix)
T = np.random.randint(1, 20, size=(n_nodes, n_nodes))
np.fill_diagonal(T, 0)

# Matrix ความเสี่ยงที่ตรวจพบ (Risk Matrix)
R = np.random.randint(1, 10, size=(n_nodes, n_nodes))
np.fill_diagonal(R, 0)

# === STEP 2: กำหนดพารามิเตอร์ของ ACO ===
n_ants = 10
n_iterations = 30
alpha = 1      # น้ำหนักของ pheromone
beta = 2       # น้ำหนักของ heuristic (risk/time)
rho = 0.1      # อัตราการระเหยของ pheromone
Q = 100        # ค่าคงที่สำหรับอัปเดต pheromone

# เริ่มต้นค่า pheromone เป็น 1
pheromone = np.ones((n_nodes, n_nodes))

# สำหรับเก็บเส้นทางที่ดีที่สุด
best_path = None
best_fitness = -np.inf
best_history = []

# === STEP 3: ฟังก์ชันฟิตเนส ===
def fitness_function(path):
    total_time = 0
    total_risk = 0
    for i in range(len(path)-1):
        a, b = path[i], path[i+1]
        total_time += T[a][b]
        total_risk += R[a][b]
    # กลับมาที่จุดเริ่มต้น
    total_time += T[path[-1]][path[0]]
    total_risk += R[path[-1]][path[0]]
    return total_risk - 0.5 * total_time

# === STEP 4: เริ่มรอบของ ACO ===
for iteration in range(n_iterations):
    all_paths = []
    all_fitness = []

    for ant in range(n_ants):
        unvisited = list(range(n_nodes))
        current = random.choice(unvisited)
        path = [current]
        unvisited.remove(current)

        while unvisited:
            probabilities = []
            for j in unvisited:
                tau = pheromone[current][j] ** alpha
                eta = (R[current][j] / (T[current][j] + 1e-6)) ** beta
                probabilities.append(tau * eta)
            probabilities = np.array(probabilities)
            probabilities /= probabilities.sum()
            next_node = random.choices(unvisited, weights=probabilities)[0]
            path.append(next_node)
            unvisited.remove(next_node)
            current = next_node

        all_paths.append(path)
        fit = fitness_function(path)
        all_fitness.append(fit)

        if fit > best_fitness:
            best_fitness = fit
            best_path = path

    # === STEP 5: อัปเดต pheromone ===
    pheromone *= (1 - rho)
    for path in all_paths:
        for i in range(len(path)-1):
            a, b = path[i], path[i+1]
            pheromone[a][b] += Q / (1 + T[a][b])
        a, b = path[-1], path[0]
        pheromone[a][b] += Q / (1 + T[a][b])

    best_history.append(best_fitness)

# === STEP 6: แสดงผลลัพธ์ ===
print("✅ Best Path:", best_path)
print("✅ Best Fitness Score:", round(best_fitness, 2))

# === STEP 7: วาดกราฟความคืบหน้า ===
plt.figure(figsize=(10, 5))
plt.plot(range(1, n_iterations+1), best_history, marker='o')
plt.title('ACO Progress: Best Fitness Over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Fitness Score')
plt.grid(True)
plt.tight_layout()
plt.show()
