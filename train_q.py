"""
Tabular Q-learning on SnakeEnv, with reward presets designed to show reward hacking.

Binary features keep the state space small enough for a plain Q-table:
2^11 = 2,048 states with basic vision, 2^14 = 16,384 with look-ahead vision.
That keeps the focus on reward design instead of neural-network debugging.

Usage:
    python train_q.py                              # train every preset and compare
    python train_q.py --preset suicidal            # train one preset
    python train_q.py --vision lookahead           # agent can detect dead ends
    python train_q.py --preset baseline --watch    # then watch one greedy game
"""

import argparse
import time

import numpy as np

from snake_env import SnakeEnv

# Each preset: (reward overrides, starve_limit, what you should expect)
PRESETS = {
    "baseline": (
        {"food": 10, "death": -10},
        100,
        "Sensible: chase food, avoid walls.",
    ),
    "suicidal": (
        {"food": 10, "death": -5, "step": -3.0},
        100,
        "Each step costs 3 and dying costs only 5, so crashing beats walking ~9 steps to food.",
    ),
    "survivor": (
        {"food": 0, "death": -10, "step": 1.0},
        None,
        "Paid just for staying alive, with no starvation limit: it circles forever and ignores food.",
    ),
    "orbiter": (
        {"food": 1, "death": -10, "closer": 1.0, "farther": 0.0},
        100,
        "Paid for approaching food but not penalized for leaving: it hovers near the food instead of eating.",
    ),
    "careful": (
        {"food": 10, "death": -50},
        100,
        "Dying costs 5x a meal, so the agent learns to avoid risky shortcuts. Best with look-ahead vision.",
    ),
    "shaped": (
        {"food": 10, "death": -10, "closer": 0.1, "farther": -0.15},
        100,
        "Well-designed shaping: leaving costs slightly more than approaching earns.",
    ),
}

MAX_STEPS = 1000  # hard cap per episode, so the "survivor" preset terminates


def obs_to_state(obs):
    """Pack the binary features into one integer index (0 .. 2^n_features - 1)."""
    return int(np.dot(obs.astype(int), 1 << np.arange(len(obs))))


def train(rewards, starve_limit, episodes=3000, alpha=0.1, gamma=0.95,
          eps_start=1.0, eps_end=0.01, seed=0, vision="basic"):
    env = SnakeEnv(rewards=rewards, starve_limit=starve_limit, vision=vision)
    rng = np.random.default_rng(seed)
    Q = np.zeros((2 ** env.observation_space.shape[0], env.action_space.n))
    eps_decay = (eps_end / eps_start) ** (1 / (episodes * 0.8))
    eps = eps_start

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        s = obs_to_state(obs)
        for _ in range(MAX_STEPS):
            a = rng.integers(3) if rng.random() < eps else int(np.argmax(Q[s]))
            obs, r, term, trunc, _ = env.step(a)
            s2 = obs_to_state(obs)
            target = r if term else r + gamma * np.max(Q[s2])
            Q[s, a] += alpha * (target - Q[s, a])
            s = s2
            if term or trunc:
                break
        eps = max(eps_end, eps * eps_decay)
    return Q


def evaluate(Q, rewards, starve_limit, episodes=200, seed=10_000, vision="basic"):
    """Greedy play. Reports the true objective (food eaten), not the shaped return."""
    env = SnakeEnv(rewards=rewards, starve_limit=starve_limit, vision=vision)
    scores, lengths, returns, how = [], [], [], {"crash": 0, "starved": 0, "timeout": 0}
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        total, info = 0.0, {"score": 0}
        for t in range(1, MAX_STEPS + 1):
            obs, r, term, trunc, info = env.step(int(np.argmax(Q[obs_to_state(obs)])))
            total += r
            if term or trunc:
                how["crash" if term else "starved"] += 1
                break
        else:
            how["timeout"] += 1
        scores.append(info["score"])
        lengths.append(t)
        returns.append(total)
    return {
        "food": np.mean(scores),
        "max_food": int(np.max(scores)),
        "steps": np.mean(lengths),
        "return": np.mean(returns),
        **{k: v / episodes for k, v in how.items()},
    }


def watch(Q, rewards, starve_limit, seed=123, delay=0.08, vision="basic"):
    env = SnakeEnv(rewards=rewards, starve_limit=starve_limit, vision=vision)
    obs, _ = env.reset(seed=seed)
    for t in range(MAX_STEPS):
        print("\033[2J\033[H" + env.render())
        obs, r, term, trunc, info = env.step(int(np.argmax(Q[obs_to_state(obs)])))
        print(f"step {t + 1}  reward {r:+.2f}  food {info['score']}")
        time.sleep(delay)
        if term or trunc:
            print(f"crashed into {info['death']}" if term else "starved")
            break


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=PRESETS, help="train one preset (default: all)")
    p.add_argument("--episodes", type=int, default=3000)
    p.add_argument("--vision", choices=["basic", "lookahead"], default="basic",
                   help="lookahead adds 3 dead-end (trap) features")
    p.add_argument("--watch", action="store_true", help="watch a greedy game after training")
    args = p.parse_args()

    names = [args.preset] if args.preset else list(PRESETS)
    print(f"{'preset':<10} {'food/game':>9} {'max':>4} {'steps':>7} {'return':>8} "
          f"{'crash':>6} {'starve':>7} {'timeout':>8}")
    for name in names:
        rewards, starve, note = PRESETS[name]
        Q = train(rewards, starve, episodes=args.episodes, vision=args.vision)
        m = evaluate(Q, rewards, starve, vision=args.vision)
        print(f"{name:<10} {m['food']:>9.2f} {m['max_food']:>4} {m['steps']:>7.1f} {m['return']:>8.1f} "
              f"{m['crash']:>6.0%} {m['starved']:>7.0%} {m['timeout']:>8.0%}")
        print(f"{'':<10} -> {note}")
        if args.watch:
            watch(Q, rewards, starve, vision=args.vision)


if __name__ == "__main__":
    main()
