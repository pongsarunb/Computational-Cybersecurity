import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parameters
a = 1.0       # ประสิทธิภาพการโจมตี
b = 1.0       # ประสิทธิภาพการป้องกัน
alpha = 0.5   # น้ำหนักต้นทุนของผู้โจมตี
beta = 0.5    # น้ำหนักต้นทุนของผู้ป้องกัน
T = 10.0      # ระยะเวลาเกม
x0 = 0.3      # สถานะเริ่มต้นของการควบคุมระบบ (โดยผู้โจมตี)

# เวลาที่ใช้สำหรับการแสดงผล
t_eval = np.linspace(0, T, 100)

# สมการพลวัตของเกม (System dynamics)
def differential_game(t, y):
    x, lam = y
    # กลยุทธ์ที่ดีที่สุดของแต่ละฝ่าย
    u = (a * lam * (1 - x)) / alpha
    v = (b * lam * x) / beta
    # สมการการเปลี่ยนแปลงสถานะและ lambda
    dxdt = a * u * (1 - x) - b * v * x
    dlamdt = - (1 - lam * (a * u + b * v))
    return [dxdt, dlamdt]

# เงื่อนไขเริ่มต้น: x(0), lambda(0)
y0 = [x0, 0.0]

# ใช้ scipy ในการแก้สมการ ODE
sol = solve_ivp(differential_game, [0, T], y0, t_eval=t_eval, method='RK45')

# แยกค่าออกมาจากผลลัพธ์
x_vals = sol.y[0]
lambda_vals = sol.y[1]

# คำนวณ u(t), v(t)
u_vals = (a * lambda_vals * (1 - x_vals)) / alpha
v_vals = (b * lambda_vals * x_vals) / beta

# วาดกราฟแสดงผล
plt.figure(figsize=(10, 6))
plt.plot(sol.t, x_vals, label='x(t): ระดับการควบคุมของผู้โจมตี')
plt.plot(sol.t, u_vals, label='u(t): ความพยายามของผู้โจมตี')
plt.plot(sol.t, v_vals, label='v(t): ความพยายามของผู้ป้องกัน')
plt.xlabel('เวลา (t)')
plt.ylabel('ค่า')
plt.title('การจำลองเกมดิฟเฟอเรนเชียล: ความมั่นคงปลอดภัยไซเบอร์')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
