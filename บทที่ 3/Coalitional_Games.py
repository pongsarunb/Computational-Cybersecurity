import itertools

# รายชื่อผู้เล่น
players = ['A', 'B', 'C']

# ฟังก์ชันค่าประโยชน์ของแต่ละกลุ่ม (Characteristic function)
value = {
    frozenset(['A']): 10,
    frozenset(['B']): 15,
    frozenset(['C']): 20,
    frozenset(['A', 'B']): 40,
    frozenset(['A', 'C']): 50,
    frozenset(['B', 'C']): 55,
    frozenset(['A', 'B', 'C']): 80,
    frozenset(): 0
}

def shapley_value(players, value):
    from math import factorial
    n = len(players)
    shapley = dict((player, 0) for player in players)

    for perm in itertools.permutations(players):
        coalition = frozenset()
        for i, player in enumerate(perm):
            prev_value = value.get(coalition, 0)
            coalition = coalition.union([player])
            new_value = value.get(coalition, 0)
            marginal_contribution = new_value - prev_value
            shapley[player] += marginal_contribution / factorial(n)
    
    return shapley

# เรียกใช้ฟังก์ชัน
shapley_vals = shapley_value(players, value)

print("\n")

# แสดงผล
for player, val in shapley_vals.items():
    print(f"Shapley Value ของ {player}: {val:.2f}")
