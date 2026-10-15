"""
Snake RL Lab: a tkinter app for watching a Q-learning agent learn Snake.

Modes
    Train AI      the agent plays with exploration and updates its Q-table every step
    Watch AI      the agent plays its best-known moves (no exploration, no learning)
    Play yourself arrow keys / WASD; same board, same rewards

Run:   python snake_rl_gui.py
Needs: snake_env.py, snake_renderer.py and train_q.py in the same folder; pip install numpy gymnasium
Keys:  Space = start/pause, arrows or WASD = steer in "Play yourself" mode
Mouse: click any empty cell to drop the apple there
"""

import time
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np

from snake_env import SnakeEnv
from snake_renderer import SnakeRenderer
from train_q import MAX_STEPS, PRESETS, obs_to_state

# ---------------- look ----------------
GRID = 10
CELL = 38
BOARD = GRID * CELL
CHART_H = 150
PANEL_W = 320

BG = "#111318"
PANEL = "#1a1d24"
TILE_A = "#171a20"
TILE_B = "#1b1e25"
LINE = "#2a2f3a"
TEXT = "#e6e8eb"
MUTED = "#8a919e"
ACCENT = "#5aa9ff"
GOOD = "#5fd38d"
BAD = "#ff6b6b"
FOOD = "#ff6b6b"

FONT = ("Segoe UI", 10)
FONT_B = ("Segoe UI", 10, "bold")
FONT_BIG = ("Segoe UI", 18, "bold")
FONT_SMALL = ("Segoe UI", 8)

ACTIONS = ["Straight", "Turn right", "Turn left"]
KEY_DIRS = {"Right": 0, "d": 0, "Down": 1, "s": 1, "Left": 2, "a": 2, "Up": 3, "w": 3}
MODES = [("train", "Train AI"), ("watch", "Watch AI"), ("human", "Play yourself")]


# ---------------- agent ----------------
class QAgent:
    def __init__(self, n_features=11, alpha=0.1, gamma=0.95, eps_decay=0.995, eps_min=0.01):
        self.Q = np.zeros((2 ** n_features, 3))
        self.alpha, self.gamma = alpha, gamma
        self.eps, self.eps_decay, self.eps_min = 1.0, eps_decay, eps_min
        self.rng = np.random.default_rng()

    def act(self, s, explore):
        if explore and self.rng.random() < self.eps:
            return int(self.rng.integers(3)), True
        return int(np.argmax(self.Q[s])), False

    def learn(self, s, a, r, s2, terminal):
        target = r if terminal else r + self.gamma * self.Q[s2].max()
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])

    def end_episode(self):
        self.eps = max(self.eps_min, self.eps * self.eps_decay)


# ---------------- app ----------------
class SnakeRLApp:
    def __init__(self, root):
        self.root = root
        root.title("Snake RL Lab")
        root.configure(bg=BG)
        root.resizable(False, False)

        self.mode = tk.StringVar(value="train")
        self.preset = tk.StringVar(value="careful")
        self.lookahead = tk.BooleanVar(value=True)
        self.speed = tk.IntVar(value=15)
        self.turbo = tk.BooleanVar(value=False)
        self.running = False
        self.human_dir = None
        self.chart_dirty = True
        self._current_preset = self.preset.get()
        self.dead = None            # snapshot of a crashed snake, shown briefly
        self.prev_snake = None      # body before the last move, for smooth interpolation
        self.grew = False
        self.last_step_time = time.perf_counter()
        self.last_chart_time = 0.0

        self._build_ui()
        self.new_agent()
        root.bind("<Key>", self.on_key)
        root.protocol("WM_DELETE_WINDOW", root.destroy)
        self.root.after(50, self.loop)

    # ---------- layout ----------
    def _build_ui(self):
        left = tk.Frame(self.root, bg=BG)
        left.grid(row=0, column=0, padx=(16, 8), pady=16, sticky="n")
        right = tk.Frame(self.root, bg=BG, width=PANEL_W)
        right.grid(row=0, column=1, padx=(8, 16), pady=16, sticky="n")

        # header above board
        tk.Label(left, text="Snake RL Lab", font=FONT_BIG, fg=TEXT, bg=BG, anchor="w").pack(fill="x")
        self.status = tk.Label(left, text="Press Space to start  ·  click the board to drop the apple",
                               font=FONT, fg=MUTED,
                               bg=BG, anchor="w", width=1)   # width=1 + fill: never widens the window
        self.status.pack(fill="x")

        self.board = tk.Canvas(left, width=BOARD, height=BOARD, bg=BG,
                               highlightthickness=1, highlightbackground=LINE)
        self.board.pack(pady=(8, 10))
        self.renderer = SnakeRenderer(self.board, GRID, CELL)
        self.board.bind("<Button-1>", self.on_board_click)
        self.board.bind("<Motion>", self.on_board_hover)
        self.board.bind("<Leave>", lambda e: setattr(self.renderer, "hover", None))

        tk.Label(left, text="Learning curve  (food per training episode)", font=FONT_B,
                 fg=TEXT, bg=BG, anchor="w").pack(fill="x")
        self.chart = tk.Canvas(left, width=BOARD, height=CHART_H, bg=PANEL,
                               highlightthickness=0)
        self.chart.pack(pady=(4, 0))

        # --- controls card
        c = self._card(right, "Controls")
        row = tk.Frame(c, bg=PANEL)
        row.pack(fill="x", pady=(0, 6))
        tk.Label(row, text="Reward preset", font=FONT, fg=MUTED, bg=PANEL).pack(side="left")
        om = tk.OptionMenu(row, self.preset, *PRESETS, command=self.on_preset)
        om.configure(bg=BG, fg=TEXT, activebackground=LINE, activeforeground=TEXT,
                     highlightthickness=0, bd=0, font=FONT, width=10)
        om["menu"].configure(bg=PANEL, fg=TEXT, activebackground=ACCENT, font=FONT)
        om.pack(side="right")
        self.preset_note = tk.Label(c, text="", font=FONT_SMALL, fg=MUTED, bg=PANEL,
                                    wraplength=PANEL_W - 30, justify="left", anchor="w")
        self.preset_note.pack(fill="x", pady=(0, 4))
        tk.Checkbutton(c, text="Look-ahead vision (detect dead ends)", variable=self.lookahead,
                       command=self.on_vision, font=FONT, fg=TEXT, bg=PANEL, selectcolor=BG,
                       activebackground=PANEL, activeforeground=TEXT, highlightthickness=0
                       ).pack(anchor="w", pady=(0, 8))

        modes = tk.Frame(c, bg=PANEL)
        modes.pack(fill="x", pady=(0, 8))
        for val, label in MODES:
            tk.Radiobutton(modes, text=label, value=val, variable=self.mode,
                           command=self.on_mode, indicatoron=False, font=FONT,
                           bg=BG, fg=TEXT, selectcolor=ACCENT, activebackground=LINE,
                           activeforeground=TEXT, bd=0, padx=8, pady=5,
                           ).pack(side="left", expand=True, fill="x", padx=1)

        self.run_btn = self._button(c, "▶  Start  (Space)", self.toggle, primary=True)
        self.run_btn.pack(fill="x", pady=(0, 8))

        sp = tk.Frame(c, bg=PANEL)
        sp.pack(fill="x")
        tk.Label(sp, text="Speed", font=FONT, fg=MUTED, bg=PANEL).pack(side="left")
        tk.Scale(sp, from_=1, to=60, orient="horizontal", variable=self.speed,
                 showvalue=False, length=130, width=10, sliderlength=18, bg=LINE,
                 troughcolor=BG, activebackground=ACCENT, highlightthickness=0, bd=0,
                 sliderrelief="flat").pack(side="left", padx=8)
        self.speed_lbl = tk.Label(sp, text="", font=FONT, fg=TEXT, bg=PANEL, width=10, anchor="e")
        self.speed_lbl.pack(side="right")
        tk.Checkbutton(c, text="Turbo training (skip animation)", variable=self.turbo,
                       font=FONT, fg=TEXT, bg=PANEL, selectcolor=BG, activebackground=PANEL,
                       activeforeground=TEXT, highlightthickness=0).pack(anchor="w", pady=(4, 6))

        btns = tk.Frame(c, bg=PANEL)
        btns.pack(fill="x")
        for text, cmd in [("Reset agent", self.reset_clicked), ("Save", self.save), ("Load", self.load)]:
            self._button(btns, text, cmd).pack(side="left", expand=True, fill="x", padx=1)

        # --- stats card
        s = self._card(right, "Stats")
        self.stat_lbls = {}
        grid = tk.Frame(s, bg=PANEL)
        grid.pack(fill="x")
        names = ["Episode", "Exploration ε", "Score", "Best (training)",
                 "Avg last 100", "Total steps", "States seen", "Last game"]
        for i, n in enumerate(names):
            tk.Label(grid, text=n, font=FONT, fg=MUTED, bg=PANEL, anchor="w").grid(
                row=i // 2, column=(i % 2) * 2, sticky="w", pady=1)
            v = tk.Label(grid, text="-", font=FONT_B, fg=TEXT, bg=PANEL, anchor="e", width=9)
            v.grid(row=i // 2, column=(i % 2) * 2 + 1, sticky="e", padx=(4, 10))
            self.stat_lbls[n] = v

        # --- agent's view card
        v = self._card(right, "What the agent sees")
        self.feat = tk.Canvas(v, width=PANEL_W - 28, height=24 + 4 * 16, bg=PANEL, highlightthickness=0)
        self.feat.pack()
        self.feat_dots = []
        groups = [("Danger", ["ahead", "right", "left"]),
                  ("Heading", ["left", "right", "up", "down"]),
                  ("Food is", ["left", "right", "up", "down"]),
                  ("Dead end", ["ahead", "right", "left"])]
        self.trap_items = []
        for g, (title, items) in enumerate(groups):
            x0 = g * 73
            t = self.feat.create_text(x0, 7, text=title, anchor="w", fill=TEXT, font=FONT_SMALL)
            if g == 3:
                self.trap_items.append(t)
            for j, name in enumerate(items):
                y = 24 + j * 16
                dot = self.feat.create_oval(x0, y - 5, x0 + 10, y + 5, fill=BG, outline=LINE)
                lbl = self.feat.create_text(x0 + 15, y, text=name, anchor="w", fill=MUTED, font=FONT_SMALL)
                self.feat_dots.append(dot)
                if g == 3:
                    self.trap_items += [dot, lbl]

        tk.Label(v, text="Q-values  (expected future reward per move)", font=FONT_SMALL,
                 fg=MUTED, bg=PANEL, anchor="w").pack(fill="x", pady=(8, 2))
        self.qcan = tk.Canvas(v, width=PANEL_W - 28, height=3 * 22, bg=PANEL, highlightthickness=0)
        self.qcan.pack()
        self.q_items = []
        mid = 120 + (PANEL_W - 28 - 120 - 50) / 2
        self.q_mid = mid
        for i, a in enumerate(ACTIONS):
            y = i * 22 + 11
            self.qcan.create_text(0, y, text=a, anchor="w", fill=MUTED, font=FONT_SMALL)
            self.qcan.create_line(mid, y - 8, mid, y + 8, fill=LINE)
            bar = self.qcan.create_rectangle(mid, y - 6, mid, y + 6, fill=ACCENT, width=0)
            val = self.qcan.create_text(PANEL_W - 28, y, text="", anchor="e", fill=TEXT, font=FONT_SMALL)
            self.q_items.append((bar, val, y))
        self.move_lbl = tk.Label(v, text="", font=FONT_SMALL, fg=MUTED, bg=PANEL, anchor="w")
        self.move_lbl.pack(fill="x", pady=(4, 0))

    def _card(self, parent, title):
        f = tk.Frame(parent, bg=PANEL, padx=12, pady=10)
        f.pack(fill="x", pady=(0, 10))
        tk.Label(f, text=title, font=FONT_B, fg=TEXT, bg=PANEL, anchor="w").pack(fill="x", pady=(0, 6))
        return f

    def _button(self, parent, text, cmd, primary=False):
        return tk.Button(parent, text=text, command=cmd, font=FONT_B if primary else FONT,
                         bg=ACCENT if primary else BG, fg="#0b1220" if primary else TEXT,
                         activebackground=LINE, activeforeground=TEXT,
                         relief="flat", bd=0, padx=8, pady=6, cursor="hand2")

    # ---------- state ----------
    def new_agent(self):
        rewards, starve, note = PRESETS[self.preset.get()]
        vision = "lookahead" if self.lookahead.get() else "basic"
        self.env = SnakeEnv(size=GRID, rewards=rewards, starve_limit=starve, vision=vision)
        self.agent = QAgent(n_features=self.env.observation_space.shape[0])
        for item in self.trap_items:
            self.feat.itemconfig(item, state="normal" if vision == "lookahead" else "hidden")
        self.history = []          # food per training episode
        self.best = 0
        self.total_steps = 0
        self.last_game = "-"
        self.preset_note.config(text=note)
        self.chart_dirty = True
        self.dead = None
        self.new_episode()
        self.redraw()

    def new_episode(self):
        self.obs, _ = self.env.reset()
        self.s = obs_to_state(self.obs)
        self.ep_steps = 0
        self.score = 0
        self.last_action = None
        self.last_random = False
        self.human_dir = None
        self.prev_snake = None
        self.grew = False

    # ---------- game loop ----------
    def turbo_on(self):
        return self.turbo.get() and self.mode.get() == "train"

    def loop(self):
        """Runs ~60 times a second. Game moves happen at the chosen speed; frames in
        between slide the snake smoothly from its old position to its new one."""
        now = time.perf_counter()
        if self.dead and now >= self.dead["until"]:
            self.dead = None
            self.last_step_time = now

        if self.running and not self.dead:
            if self.turbo_on():
                deadline = now + 0.035
                while time.perf_counter() < deadline:
                    self.step(animate=False)
                self.prev_snake = None
                self.update_panels()
                self.maybe_chart()
                self.render(1.0)
                self.root.after(5, self.loop)
                return
            if now - self.last_step_time >= 1.0 / self.speed.get():
                self.last_step_time = now
                self.step(animate=True)
                self.update_panels()
                self.maybe_chart()

        t = 1.0
        if self.running and not self.dead and self.prev_snake is not None:
            t = min(1.0, (time.perf_counter() - self.last_step_time) * self.speed.get())
        self.render(t)
        self.root.after(15, self.loop)

    def step(self, animate=True):
        mode = self.mode.get()
        if mode == "human":
            a, rand = self.human_action(), False
        else:
            a, rand = self.agent.act(self.s, explore=(mode == "train"))
        before = list(self.env.snake)
        food_before = self.env.food
        old_score = self.score

        obs, r, term, trunc, info = self.env.step(a)
        s2 = obs_to_state(obs)
        if mode == "train":
            self.agent.learn(self.s, a, r, s2, term)
        self.last_action, self.last_random = a, rand
        self.obs, self.s = obs, s2
        self.ep_steps += 1
        self.total_steps += 1
        self.score = info["score"]
        self.prev_snake = before
        self.grew = len(self.env.snake) > len(before)
        if animate and self.score > old_score:
            self.renderer.burst(food_before)

        if term or trunc or self.ep_steps >= MAX_STEPS:
            how = ({"self": "hit itself", "wall": "hit a wall"}.get(info["death"], "crashed") if term
                   else ("starved" if trunc else "timed out"))
            self.finish_episode(how, animate)

    def finish_episode(self, how, animate=True):
        mode = self.mode.get()
        if mode == "train":
            self.history.append(self.score)
            self.best = max(self.best, self.score)
            self.agent.end_episode()
            self.chart_dirty = True
        short = {"hit itself": "self", "hit a wall": "wall", "timed out": "time"}.get(how, how)
        self.last_game = f"{self.score} ({short})"
        if animate:
            hold = float("inf") if mode == "human" else 0.55
            self.dead = {"snake": list(self.env.snake), "food": self.env.food,
                         "until": time.perf_counter() + hold}
        if mode == "human":
            self.running = False
            self.status.config(text=f"You {how} with {self.score} food. Space to retry.", fg=BAD)
            self._update_run_btn()
        self.new_episode()

    def human_action(self):
        if self.human_dir is None:
            return 0
        diff = (self.human_dir - self.env.dir_idx) % 4
        self.human_dir = None
        return {0: 0, 1: 1, 3: 2}.get(diff, 0)   # reversing is ignored

    # ---------- drawing ----------
    def redraw(self):
        self.update_panels()
        self.render(1.0)
        if self.chart_dirty:
            self.draw_chart()

    def render(self, t):
        if self.dead:
            d = self.dead
            self.renderer.render(d["snake"], None, False, 1.0, d["food"], dead=True)
        else:
            self.renderer.render(list(self.env.snake), self.prev_snake, self.grew, t, self.env.food)

    def maybe_chart(self):
        now = time.perf_counter()
        if self.chart_dirty and now - self.last_chart_time > (0.5 if self.turbo_on() else 0.15):
            self.last_chart_time = now
            self.draw_chart()

    def update_panels(self):
        a = self.agent
        h = self.history
        st = self.stat_lbls
        st["Episode"].config(text=str(len(h)))
        st["Exploration ε"].config(text=f"{a.eps:.3f}")
        st["Score"].config(text=str(self.score))
        st["Best (training)"].config(text=str(self.best))
        st["Avg last 100"].config(text=f"{np.mean(h[-100:]):.2f}" if h else "-")
        st["Total steps"].config(text=f"{self.total_steps:,}")
        st["States seen"].config(text=f"{int(np.any(a.Q != 0, axis=1).sum()):,} / {len(a.Q):,}")
        st["Last game"].config(text=self.last_game)
        self.speed_lbl.config(text="turbo" if self.turbo_on() else f"{self.speed.get()} moves/s")

        for i, (dot, on) in enumerate(zip(self.feat_dots, self.obs)):
            col = (BAD if i < 3 else ACCENT if i < 7 else FOOD if i < 11 else "#ffb547") if on else BG
            self.feat.itemconfig(dot, fill=col)

        q = a.Q[self.s]
        scale = max(1e-6, np.abs(q).max())
        half = (PANEL_W - 28 - 120 - 50) / 2
        best = int(np.argmax(q))
        for i, (bar, val, y) in enumerate(self.q_items):
            w = q[i] / scale * half
            self.qcan.coords(bar, self.q_mid, y - 6, self.q_mid + w, y + 6)
            self.qcan.itemconfig(bar, fill=GOOD if (i == best and q.any()) else (BAD if q[i] < 0 else ACCENT))
            self.qcan.itemconfig(val, text=f"{q[i]:+.2f}")

        if self.last_action is None:
            self.move_lbl.config(text="Next move: best Q-value (green), or a random move while exploring")
        else:
            how = "random (exploring)" if self.last_random else (
                "your key" if self.mode.get() == "human" else "highest Q-value")
            self.move_lbl.config(text=f"Last move: {ACTIONS[self.last_action]}  ←  {how}")

    def draw_chart(self):
        self.chart_dirty = False
        c = self.chart
        c.delete("all")
        W, H, L, B, T = BOARD, CHART_H, 34, 20, 10
        h = np.array(self.history, dtype=float)
        if len(h) < 2:
            c.create_text(W / 2, H / 2, text="Train the agent to see its learning curve",
                          fill=MUTED, font=FONT)
            return
        ymax = max(5, h.max())
        n = len(h)

        def px(i):
            return L + (W - L - 10) * i / (n - 1)

        def py(v):
            return H - B - (H - B - T) * v / ymax

        for frac in (0, 0.5, 1):
            y = py(ymax * frac)
            c.create_line(L, y, W - 10, y, fill=LINE)
            c.create_text(L - 6, y, text=f"{ymax * frac:.0f}", anchor="e", fill=MUTED, font=FONT_SMALL)
        c.create_text(L, H - 6, text="0", anchor="w", fill=MUTED, font=FONT_SMALL)
        c.create_text(W - 10, H - 6, text=f"{n} episodes", anchor="e", fill=MUTED, font=FONT_SMALL)

        step = max(1, n // 400)
        for i in range(0, n, step):
            x, y = px(i), py(h[i])
            c.create_rectangle(x, y, x + 1, y + 1, outline="", fill="#3a4a63")
        win = min(50, n)
        avg = np.convolve(h, np.ones(win) / win, mode="valid")
        idx = np.arange(win - 1, n)
        pts = []
        for i in range(0, len(avg), step):
            pts += [px(idx[i]), py(avg[i])]
        if len(pts) >= 4:
            c.create_line(*pts, fill=ACCENT, width=2, smooth=True)
        c.create_text(L + 6, T, text=f"— {win}-episode average", anchor="nw", fill=ACCENT, font=FONT_SMALL)

    # ---------- controls ----------
    def toggle(self):
        self.running = not self.running
        if self.running:
            self.dead = None
            self.last_step_time = time.perf_counter()
            hint = {"train": "Training: exploring and learning",
                    "watch": "Watching the agent's best moves",
                    "human": "Your turn: arrows or WASD"}[self.mode.get()]
            self.status.config(text=hint, fg=GOOD)
        else:
            self.status.config(text="Paused", fg=MUTED)
        self._update_run_btn()

    def _update_run_btn(self):
        self.run_btn.config(text="❚❚  Pause  (Space)" if self.running else "▶  Start  (Space)")

    def on_mode(self):
        self.dead = None
        self.new_episode()
        self.last_game = "-"
        if self.mode.get() == "human" and self.speed.get() > 12:
            self.speed.set(8)
        self.redraw()
        if self.running:
            self.toggle(); self.toggle()   # refresh status text

    def on_preset(self, _value):
        if self.history and not messagebox.askyesno(
                "Change reward preset", "Changing the reward resets the agent. Continue?"):
            self.preset.set(self._current_preset)
            return
        self._current_preset = self.preset.get()
        self.new_agent()

    def on_vision(self):
        if self.history and not messagebox.askyesno(
                "Change agent vision", "Changing what the agent sees resets it. Continue?"):
            self.lookahead.set(not self.lookahead.get())
            return
        self.new_agent()

    def reset_clicked(self):
        self.new_agent()
        self.status.config(text="Agent reset: it knows nothing again", fg=MUTED)

    def on_board_click(self, e):
        """Drop the apple anywhere: the agent re-plans from its new observation."""
        x, y = int(e.x // CELL), int(e.y // CELL)
        if not (0 <= x < GRID and 0 <= y < GRID) or (x, y) in self.env.snake or self.dead:
            return
        self.env.food = (x, y)
        self.obs = self.env._obs()
        self.s = obs_to_state(self.obs)
        self.renderer.ping((x, y))
        self.update_panels()

    def on_board_hover(self, e):
        x, y = int(e.x // CELL), int(e.y // CELL)
        self.renderer.hover = (x, y) if (0 <= x < GRID and 0 <= y < GRID) else None

    def on_key(self, e):
        if e.keysym == "space":
            self.toggle()
        elif e.keysym in KEY_DIRS and self.mode.get() == "human":
            self.human_dir = KEY_DIRS[e.keysym]
            if not self.running:
                self.toggle()

    def save(self):
        path = filedialog.asksaveasfilename(defaultextension=".npz", initialfile="snake_agent.npz",
                                            filetypes=[("Agent", "*.npz")])
        if path:
            np.savez(path, Q=self.agent.Q, eps=self.agent.eps, preset=self.preset.get(),
                     history=np.array(self.history), lookahead=self.lookahead.get())
            self.status.config(text="Agent saved", fg=GOOD)

    def load(self):
        path = filedialog.askopenfilename(filetypes=[("Agent", "*.npz")])
        if not path:
            return
        d = np.load(path, allow_pickle=False)
        self.preset.set(str(d["preset"]))
        self._current_preset = self.preset.get()
        self.lookahead.set(bool(d["lookahead"]) if "lookahead" in d else d["Q"].shape[0] > 2048)
        self.new_agent()
        self.agent.Q = d["Q"]
        self.agent.eps = float(d["eps"])
        self.history = [int(x) for x in d["history"]]
        self.best = max(self.history, default=0)
        self.chart_dirty = True
        self.redraw()
        self.status.config(text="Agent loaded", fg=GOOD)


def main():
    root = tk.Tk()
    SnakeRLApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
