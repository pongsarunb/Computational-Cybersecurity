from itertools import product

# Define player types and their probabilities
attacker_types = ["novice", "pro"]
type_prob = {"novice": 0.6, "pro": 0.4}

# Define actions
attacker_actions = ["A1", "A2"]
defender_actions = ["D1", "D2"]

# Payoff matrix: (attacker_type, attacker_action, defender_action): (attacker_payoff, defender_payoff)
payoff_matrix = {
    ("novice", "A1", "D1"): (2, -2),
    ("novice", "A1", "D2"): (0, -1),
    ("novice", "A2", "D1"): (3, -3),
    ("novice", "A2", "D2"): (1, -1.5),
    ("pro", "A1", "D1"): (3, -3),
    ("pro", "A1", "D2"): (2, -2),
    ("pro", "A2", "D1"): (4, -4),
    ("pro", "A2", "D2"): (3, -2.5),
}

# Helper function to calculate defender's expected utility
def expected_defender_payoff(attacker_strategy, defender_action):
    expected = 0
    for a_type in attacker_types:
        a_action = attacker_strategy[a_type]
        _, d_payoff = payoff_matrix[(a_type, a_action, defender_action)]
        expected += type_prob[a_type] * d_payoff
    return expected

# Search for Bayesian Nash Equilibrium
equilibria = []

# Try every combination of attacker strategies (for both types) and defender strategy
for attacker_strat in product(attacker_actions, repeat=2):  # (novice_action, pro_action)
    strategy_dict = {"novice": attacker_strat[0], "pro": attacker_strat[1]}

    for defender_action in defender_actions:
        # Check if attacker of each type is best responding
        best_response = True
        for a_type in attacker_types:
            opponent_action = defender_action
            current_action = strategy_dict[a_type]
            current_payoff, _ = payoff_matrix[(a_type, current_action, opponent_action)]

            # Check all alternative actions
            for alt_action in attacker_actions:
                alt_payoff, _ = payoff_matrix[(a_type, alt_action, opponent_action)]
                if alt_payoff > current_payoff:
                    best_response = False
                    break
            if not best_response:
                break

        # Check if defender is best responding
        current_defender_payoff = expected_defender_payoff(strategy_dict, defender_action)
        alt_defender_better = False
        for alt_defender_action in defender_actions:
            if alt_defender_action == defender_action:
                continue
            alt_payoff = expected_defender_payoff(strategy_dict, alt_defender_action)
            if alt_payoff > current_defender_payoff:
                alt_defender_better = True
                break

        if best_response and not alt_defender_better:
            equilibria.append((strategy_dict.copy(), defender_action))

# Output results
print("\nBayesian Nash Equilibria Found:")
for eq in equilibria:
    print(f"Attacker strategy: {eq[0]}, Defender strategy: {eq[1]}")
