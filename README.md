## Introduction to Reinforcement Learning — Examples and Exercises

This repository contains examples and exercises for an **Introductory course on Reinforcement Learning (RL)**, based on:

- *Reinforcement Learning: An Introduction* by Sutton and Barto (2018 Edition)
- *Artificial Intelligence* by Russell Norvig (2022 edition)
- *David Silver’s Reinforcement Learning Course (UCL / DeepMind)*

---

## Overview

The notebooks in this repository are designed for **educational purposes**.  
They aim to **illustrate and compare** different reinforcement learning approaches rather than achieve state-of-the-art results.

All examples are implemented using:

- **[Gymnasium](https://gymnasium.farama.org/)** (Farama Foundation) for environment interaction  
- **[Keras](https://keras.io/)** and occasionally **[PyTorch](https://pytorch.org/)** for deep learning components

---

## Repository Structure

### Core Concepts

| Folder / Notebook | Description |
|-------------------|-------------|
| `00`  | Basic environment creation and setup |
| `015` | Dynamic Programming |
| `018` | Monte Carlo methods |
| `019` | TD(0) algorithms |
| `020` | SARSA |
| `021` | Q-Learning |

---

### Value-Based Methods

| Notebook | Method |
|-----------|---------|
| `021` | Q-Learning |
| `030` | DQN — Deep Q-Network |
| `031` | DDQN — Double Deep Q-Network |
| `032` | D3QN — Dueling Q-Network |
| `033` | PER — Prioritized Experience Replay |

---

### Policy-Based Methods

| Notebook | Method |
|-----------|---------|
| `035` | PPO — Proximal Policy Optimization |
| `040` | A2C — Advantage Actor-Critic |
| `050` | REINFORCE |
| `060` | TD3 — Twin Delayed DDPG |
| `070` | SAC — Soft Actor-Critic |
| `080` | Heuristic approaches |
| `090` | LLM-based Agents |
| `100` | Torch / Keras Neural Network examples |

---

### Stable Baselines 3 Implementations

| Notebook | Method | Original Paper | Year |
|-----------|---------|---------|---------|
| `130` | DQN (SB3) — Deep Q-Network | https://arxiv.org/abs/1312.5602 | 2013 |
| `135` | PPO (SB3) — Proximal Policy Optimization | https://arxiv.org/abs/1707.06347 | 2017 |
| `140` | A2C (SB3) — Advantage Actor-Critic |https://arxiv.org/abs/1602.01783| 2016 |
| `160` | TD3 (SB3) — Twin Critics | https://arxiv.org/pdf/1802.09477 | 2018 |
| `170` | SAC (SB3) — Soft Actor-Critic | https://arxiv.org/abs/1801.01290 | 2018 |

---

### Distributional Methods with Stable Baselines 3 

| Notebook | Method |  Original Paper |  Year |
|-----------|---------|---------|---------|
| `210` | TQC (SB3) — Truncate Quantile Critic | https://arxiv.org/pdf/2005.04269 | 2020 |

---

## License

All notebooks are released under the **MIT License**, unless otherwise stated in the code.  
Some examples may be adapted or inspired by existing open-source material, with appropriate attribution.

---

## Notes

- The focus is on clarity and understanding rather than benchmark performance.  
- Each notebook is self-contained and can be executed independently.  
- Intended as a practical complement to the theoretical content of the course.

