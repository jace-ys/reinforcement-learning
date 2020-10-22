#!/usr/bin/env python
# coding: utf-8

CID = "1055349"

states = []
rewards = []

for digit in CID:
    states.append((int(digit) + 2) % 4)
    rewards.append(int(digit) % 4)

print("The observed trace of states and rewards:", end=" ")
for idx, value in enumerate(states):
    print(f"s{value}", end=" ")
    print(rewards[idx], end=" ")


V = [0, 0, 0, 0]
alpha = 1
gamma = 1

for i in range(len(states)-1):
    current = states[i]
    next = states[i+1]
    reward = rewards[i]
    V[current] += alpha * (reward + gamma * V[next] - V[current])

first = states[0]
print(f"The estimated value of the first state (s{first}) in the trace: {V[first]}")




