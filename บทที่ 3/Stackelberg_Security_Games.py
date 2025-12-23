# Define targets
targets = {
    'A': {'attack_loss': 10, 'defense_loss': 2},
    'B': {'attack_loss': 8, 'defense_loss': 3},
    'C': {'attack_loss': 6, 'defense_loss': 2}
}

# Solve the system of equations
# Expected Loss when attacking a target = (1 - p) * attack_loss + p * defense_loss

# Let pA, pB, pC be the probability to defend A, B, C respectively
# Need to solve:
# 10 - 8pA = 8 - 5pB = 6 - 4pC
# and pA + pB + pC = 1

from sympy import symbols, Eq, solve

# Define variables
pA, pB, pC = symbols('pA pB pC')

# Define equations based on indifference conditions
eq1 = Eq(10 - 8*pA, 8 - 5*pB)
eq2 = Eq(8 - 5*pB, 6 - 4*pC)
eq3 = Eq(pA + pB + pC, 1)

# Solve the system
solution = solve((eq1, eq2, eq3), (pA, pB, pC))

# Extract probabilities
pA_val = solution[pA]
pB_val = solution[pB]
pC_val = solution[pC]

# Print defense strategy
print(f"\nDefense Strategy:")
print(f"Protect A: {pA_val:.4f} ({pA_val*100:.2f}%)")
print(f"Protect B: {pB_val:.4f} ({pB_val*100:.2f}%)")
print(f"Protect C: {pC_val:.4f} ({pC_val*100:.2f}%)")

# Compute expected attacker's payoff
expected_loss_A = 10 - 8 * pA_val
expected_loss_B = 8 - 5 * pB_val
expected_loss_C = 6 - 4 * pC_val

print(f"\nExpected Attacker's Payoff for each target:")
print(f"Target A: {expected_loss_A:.4f}")
print(f"Target B: {expected_loss_B:.4f}")
print(f"Target C: {expected_loss_C:.4f}")

# Since the attacker is indifferent (expected payoffs are equal), print common value

expected_attacker_payoff = expected_loss_A  # (or expected_loss_B, or expected_loss_C)

print(f"\nAttacker's Expected Payoff: {expected_attacker_payoff:.4f}")
print(f"Defender's Expected Loss: {expected_attacker_payoff:.4f}")
print("\n")
