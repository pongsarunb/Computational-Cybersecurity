import numpy as np
import matplotlib.pyplot as plt

# ตั้งค่าขอบเขตเวลา
T = 50              # ระยะเวลาในการจำลอง
dt = 0.01           # ขนาดช่วงเวลา
steps = int(T / dt)

# สร้างตัวแปรเก็บค่าเวลา และสัดส่วนของกลยุทธ์ S (secure behavior)
time = np.linspace(0, T, steps)
x = np.zeros(steps)  # สัดส่วนของพฤติกรรม S
x[0] = 0.3           # จุดเริ่มต้น: 30% ของพนักงานมีพฤติกรรมปลอดภัย

# Replicator Dynamics: dx/dt = x(2x^2 - 3x + 1)
def dx_dt(x):
    return x * (2 * x**2 - 3 * x + 1)

# จำลองการเปลี่ยนแปลงของสัดส่วน x ตามเวลา
for t in range(1, steps):
    x[t] = x[t - 1] + dx_dt(x[t - 1]) * dt
    # จำกัดให้อยู่ระหว่าง 0 และ 1
    x[t] = max(0, min(1, x[t]))

# วาดกราฟ
plt.figure(figsize=(10, 6))
plt.plot(time, x, label='Proportion of Secure Behavior (S)', color='green')
plt.axhline(y=0.5, color='red', linestyle='--', label='Unstable equilibrium (x=0.5)')
plt.axhline(y=0.0, color='gray', linestyle='--')
plt.axhline(y=1.0, color='gray', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Proportion of Secure Behavior (x)')
plt.title('Evolutionary Game: Cybersecurity Behavior Dynamics')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
