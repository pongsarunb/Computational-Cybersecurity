
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ✅ 1. กำหนด States และ Actions
states = ["normal_traffic", "suspicious_traffic", "attack_detected"]
actions = ["allow", "monitor", "block"]

state_to_idx = {s: i for i, s in enumerate(states)}
action_to_idx = {a: i for i, a in enumerate(actions)}

# ✅ 2. Q-table และ Parameters
Q = np.zeros((len(states), len(actions)))
alpha = 0.1      # Learning rate
gamma = 0.9      # Discount factor
epsilon = 0.2    # Exploration rate
episodes = 500

# ✅ 3. Reward Function
def get_reward(state, action):
    if state == "normal_traffic":
        return 1 if action == "allow" else -1
    elif state == "suspicious_traffic":
        return 2 if action == "monitor" else -2
    elif state == "attack_detected":
        return 5 if action == "block" else -5
    return 0

# ✅ 4. Training Loop
rewards_per_episode = []

for episode in range(episodes):
    state = random.choice(states)
    total_reward = 0

    for step in range(10):  # จำกัดจำนวน step ต่อ episode
        s_idx = state_to_idx[state]

        # Epsilon-Greedy
        if np.random.rand() < epsilon:
            action = random.choice(actions)
        else:
            action = actions[np.argmax(Q[s_idx])]

        a_idx = action_to_idx[action]
        reward = get_reward(state, action)
        total_reward += reward

        # State Transition
        if state == "normal_traffic":
            next_state = np.random.choice(["normal_traffic", "suspicious_traffic"], p=[0.8, 0.2])
        elif state == "suspicious_traffic":
            next_state = np.random.choice(["normal_traffic", "attack_detected"], p=[0.3, 0.7])
        else:
            next_state = "normal_traffic"

        ns_idx = state_to_idx[next_state]

        # Q-Learning Update
        Q[s_idx][a_idx] = Q[s_idx][a_idx] + alpha * (
            reward + gamma * np.max(Q[ns_idx]) - Q[s_idx][a_idx]
        )

        state = next_state

    rewards_per_episode.append(total_reward)

# ✅ 5. Visualize Q-Table และ Learning Progress
q_table_df = pd.DataFrame(Q, columns=actions, index=states)

sns.set(style="whitegrid")
plt.figure(figsize=(10, 4))
plt.plot(rewards_per_episode)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Learning Progress of RL Agent in Cybersecurity")
plt.tight_layout()
plt.show()

print("Final Q-Table:")
print(q_table_df)
