# Snake RL Lab

A small lab for watching a Q-learning agent learn Snake, and for seeing how badly it behaves when you design the reward wrong.

The project has three parts: a Snake environment built on [Gymnasium](https://gymnasium.farama.org/), a command-line trainer that compares reward presets, and a tkinter desktop app where you can train the agent, watch it play or play yourself.

## Features

- **Gymnasium environment** (`SnakeEnv`) where every reward is configurable, so you can break them on purpose.
- **Tabular Q-learning.** The binary observation features keep the state space small: 2,048 states with basic vision and 16,384 with look-ahead. You can read the whole Q-table, with no neural network to debug.
- **Reward-hacking presets** that show common ways a reward can go wrong: an agent that crashes on purpose, one that loops forever and one that circles the food without eating it.
- **Look-ahead vision**: three extra features that flag moves leading into dead ends. Each check is a flood fill that also tests whether the snake can still reach its own tail.
- **Desktop app**:
  - smooth snake animation
  - a live learning curve
  - the agent's current observation and its Q-values for each move
  - click any empty cell to move the apple
  - save and load trained agents

## Requirements

- Python 3.9+
- `numpy`, `gymnasium`
- tkinter, for the GUI. It is included with the standard Python installers on Windows and macOS. On Debian/Ubuntu, install `python3-tk`.

```bash
pip install numpy gymnasium
```

## Quick start

### Desktop app

```bash
python snake_rl_gui.py
```

| Mode | What happens |
|---|---|
| **Train AI** | The agent explores (ε-greedy) and updates its Q-table every step |
| **Watch AI** | The agent plays its best known moves without exploring or learning |
| **Play yourself** | You steer with the arrow keys or WASD, on the same board with the same rewards |

**Controls:** `Space` starts and pauses. Arrow keys or `WASD` steer. Click an empty cell to drop the apple there. Tick **Turbo training** to skip the animation and train much faster.

### Command-line trainer

```bash
python train_q.py                             # train every preset and compare them
python train_q.py --preset suicidal           # train a single preset
python train_q.py --vision lookahead          # give the agent dead-end detection
python train_q.py --preset baseline --watch   # then watch one greedy game in the terminal
python train_q.py --episodes 10000            # train for longer (default 3000)
```

The trainer prints, for each preset, the average food per game, the maximum, the average episode length, the average return, and how games ended (crash, starvation or timeout).

### Environment sanity check

```bash
python snake_env.py   # runs a random agent for 200 episodes and prints an ASCII board
```

## Reward presets

| Preset | Rewards | What the agent learns |
|---|---|---|
| `baseline` | food +10, death −10 | Chases food and avoids walls |
| `suicidal` | food +10, death −5, step −3 | Crashes on purpose, because dying costs less than walking about 9 steps to the food |
| `survivor` | food 0, death −10, step +1, no starvation limit | Circles forever and ignores the food |
| `orbiter` | food +1, death −10, closer +1 | Hovers near the food instead of eating it |
| `careful` | food +10, death −50 | Avoids risky shortcuts; works best with look-ahead vision |
| `shaped` | food +10, death −10, closer +0.1, farther −0.15 | Uses shaping that helps learning without being exploitable |

To add your own preset, add an entry to `PRESETS` in `train_q.py`. The GUI picks it up automatically.

## How it works

**Observation.** There are 11 binary features, plus 3 with look-ahead vision:

| Group | Features |
|---|---|
| Danger | straight, right, left |
| Heading | left, right, up, down |
| Food direction | left, right, up, down |
| Dead end (look-ahead only) | straight, right, left |

**Actions.** The three actions are relative to the snake's heading: `0` straight, `1` turn right, `2` turn left.

**Episode end.** An episode ends when the snake hits a wall or itself, when it fills the board, or when it starves. It starves after `starve_limit × length` steps without eating. Set `starve_limit=None` to turn starvation off. Episodes are also capped at 1,000 steps.

**Learning.** The binary features are packed into one integer state index. A standard Q-learning update then runs with α = 0.1 and γ = 0.95, and ε decays from 1.0 to 0.01.

## Using the environment in your own code

```python
from snake_env import SnakeEnv

env = SnakeEnv(size=10, rewards={"food": 10, "death": -10, "step": -0.01},
               starve_limit=100, vision="lookahead")
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
print(env.render())
```

The available reward keys are `food`, `death`, `step`, `closer`, `farther` and `starve`.

## Project structure

```
snake_env.py       Gymnasium Snake environment (observations, rewards, trap detection)
train_q.py         Tabular Q-learning, reward presets, CLI evaluation and terminal viewer
snake_rl_gui.py    tkinter app: train / watch / play, learning curve, Q-value inspector
snake_renderer.py  Canvas renderer: smooth tapered snake, apple, particles and animations
```
