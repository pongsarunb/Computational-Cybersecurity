import numpy as np

# กำหนดค่าเริ่มต้น
DAMAGE = -100   # ความเสียหายจากการโจมตีสำเร็จ
MONITOR_COST = -20  # ต้นทุนการตรวจสอบ
TIME_HORIZON = 3  # จำนวนช่วงเวลา

# กำหนดสภาพแวดล้อม
STATE_NORMAL = 'Normal'
STATE_HIGH_RISK = 'High Risk'

# ความน่าจะเป็นการโจมตีสำเร็จ
success_prob = {
    STATE_NORMAL: 0.5,
    STATE_HIGH_RISK: 0.8
}

# ความน่าจะเป็นการเปลี่ยนสถานะ
def next_state(current_state, attack_success):
    if attack_success:
        return STATE_HIGH_RISK if np.random.rand() < 0.9 else STATE_NORMAL
    else:
        return STATE_NORMAL if np.random.rand() < 0.8 else STATE_HIGH_RISK

# จำลองกลยุทธ์
def simulate(strategy, trials=10000):
    total_payoff = 0

    for _ in range(trials):
        state = STATE_NORMAL
        payoff = 0

        for t in range(TIME_HORIZON):
            attacker_action = np.random.choice(['Attack', 'No Attack'])

            if strategy == 'Monitor Always':
                defender_monitor = True
            elif strategy == 'No Monitor Always':
                defender_monitor = False
            elif strategy == 'Monitor High Risk Only':
                defender_monitor = (state == STATE_HIGH_RISK)
            else:
                raise ValueError("Unknown strategy")

            if attacker_action == 'Attack':
                attack_success = np.random.rand() < success_prob[state]
                if defender_monitor:
                    payoff += MONITOR_COST  # เสียต้นทุนการตรวจสอบ
                else:
                    if attack_success:
                        payoff += DAMAGE  # เสียหายจากการโจมตีสำเร็จ
            else:
                if defender_monitor:
                    payoff += MONITOR_COST  # เสียต้นทุนการตรวจสอบ

            # อัปเดตสถานะ
            if attacker_action == 'Attack':
                attack_success = np.random.rand() < success_prob[state]
                state = next_state(state, attack_success)
            else:
                # ถ้าไม่โจมตี ให้สุ่มเปลี่ยนสถานะเล็กน้อย (Optional)
                state = STATE_NORMAL if np.random.rand() < 0.8 else STATE_HIGH_RISK

        total_payoff += payoff

    expected_payoff = total_payoff / trials
    return expected_payoff

# ทดลองกลยุทธ์ต่าง ๆ
strategies = ['Monitor Always', 'No Monitor Always', 'Monitor High Risk Only']

print("\n")
for strat in strategies:
    expected = simulate(strat)
    print(f"Strategy: {strat} => Expected Payoff: {expected:.2f}")
