# Cafe Barista — RL Practice

A custom Gymnasium environment for single-agent RL practice with SB3. The agent runs a small cafe during a morning rush: customers queue up with orders (espresso, latte, pastry), each with a patience timer and a tip that decays while they wait. Three stations, one order at a time each. The agent decides who to serve next.

It's a **scheduling problem**, not a control problem — small observation space, fast to train, but rich enough that PPO clearly beats both random and FIFO baselines.

The objective is to solve a scheduling problem using RL. After executing both programs you will see how optimizing for Tips generates the best results. Optimizing only for revenue is not good enough. Is it a learning for real life?

## Files

- `cafe_env.py` — the Gymnasium environment
- `cafe_render.py` — 8-bit pixel-art pygame renderer (`human` and `rgb_array` modes)
- `agent_cafe_ppo.ipynb` — train PPO, compare against baselines, save a GIF
- `agent_cafe_heuristics.ipynb` — use heuristics, save a GIF

## Run

```bash
pip install gymnasium stable-baselines3 pygame matplotlib pillow
jupyter notebook train_cafe.ipynb
```

## Results

After ~80k timesteps (~30s on CPU):

| Policy    | Mean reward |
|-----------|-------------|
| Random    | ~169        |
| FIFO      | ~171        |
| **PPO**   | **~203**    |

## Action & observation spaces

- **Action** (`Discrete(5)`): `0` = idle, `1..4` = serve queue slot
- **Observation** (`Box(23,)`): per-slot order one-hot + patience + present flag, plus per-station busy time