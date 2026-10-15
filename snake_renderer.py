"""
Realistic snake rendering for a tkinter Canvas.

The snake is drawn as a continuous, tapered body made of many overlapping discs
sampled along its path, in layers: ground shadow -> dark outline -> body colour ->
saddle markings -> highlights. Between grid steps the body is interpolated along the
path, so it glides instead of jumping. The head turns smoothly, its eyes track the
food, and the tongue flicks (faster when food is close).
"""

import math
import random
import time
import tkinter as tk

# ---------- palette ----------
GROUND_A = "#182019"
GROUND_B = "#1b231c"
SPECKLES = ["#1f2a20", "#222d23", "#151c16", "#26301f"]
SHADOW = "#0e130f"

BODY_HEAD = (78, 190, 98)
BODY_TAIL = (34, 118, 66)
SADDLE = (14, 52, 30)
DEAD_HEAD = (128, 136, 120)
DEAD_TAIL = (84, 92, 80)
IRIS = "#f0c93e"
TONGUE = "#e3405a"

APPLE = "#d8382f"
APPLE_SHADE = "#b02621"
APPLE_EDGE = "#6e1410"
APPLE_LIGHT = "#ff8b7c"
LEAF = "#62c04b"
LEAF_EDGE = "#2c6a22"
STEM = "#6b4423"


def rgb(c):
    return "#%02x%02x%02x" % tuple(max(0, min(255, int(v))) for v in c)


def mix(a, b, t):
    return tuple(x + (y - x) * t for x, y in zip(a, b))


def hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def smoothstep(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


# Head outline in local units of one cell (x = forward, y = sideways)
_HEAD = [(0.50, 0.0), (0.47, 0.08), (0.37, 0.16), (0.20, 0.235), (0.02, 0.265), (-0.15, 0.245),
         (-0.28, 0.16), (-0.33, 0.0), (-0.28, -0.16), (-0.15, -0.245), (0.02, -0.265),
         (0.20, -0.235), (0.37, -0.16), (0.47, -0.08)]
_MARK = [(0.24, 0.0), (0.02, 0.14), (-0.18, 0.13), (-0.04, 0.0), (-0.18, -0.13), (0.02, -0.14)]
HEAD_SHAPE = [(x * 1.0, y * 1.2) for x, y in _HEAD]   # a bit wider than the neck, like a real snake
HEAD_MARK = [(x * 1.0, y * 1.2) for x, y in _MARK]
BODY_R = 0.31        # max body radius, in cells


class SnakeRenderer:
    def __init__(self, canvas, grid, cell, seed=7):
        self.c = canvas
        self.grid = grid
        self.cell = cell
        self.particles = []     # [x, y, vx, vy, life, max_life, (r,g,b), size]
        self.popups = []        # [x, y, text, life, max_life]
        self.rings = []         # [x, y, life, max_life]
        self.hover = None
        self.tongue_phase = random.random()
        self._last = time.perf_counter()
        self._colors = {}
        for name, val in list(globals().items()):          # pin the fixed palette too
            if isinstance(val, str) and val.startswith("#") and len(val) == 7:
                self._keep(val)
        for extra in ("#10241a", "#0d1a12", "#9aa093", "#1a1d1a", "#111111", "#fffbe6", "#ffe3dc"):
            self._keep(extra)
        self.draw_ground(seed)

    # Tk frees a colour when the last item using it is deleted, and re-allocating it
    # every frame is slow. Quantise colours and pin each one with a hidden item.
    def _keep(self, hexstr):
        self.c.create_rectangle(-9, -9, -8, -8, fill=hexstr, outline=hexstr, state="hidden", tags="keep")

    def col(self, c):
        key = tuple(max(0, min(255, int(v))) // 4 * 4 for v in c)
        h = self._colors.get(key)
        if h is None:
            h = "#%02x%02x%02x" % key
            self._colors[key] = h
            self._keep(h)
        return h

    # ---------- public ----------
    def center(self, p):
        return (p[0] + 0.5) * self.cell, (p[1] + 0.5) * self.cell

    def burst(self, cell_pos):
        """Juice splash + floating +1 when food is eaten."""
        x, y = self.center(cell_pos)
        for _ in range(18):
            a = random.uniform(0, 2 * math.pi)
            v = random.uniform(50, 170)
            col = random.choice([hex_rgb(APPLE), hex_rgb(APPLE_LIGHT), (255, 214, 102), hex_rgb(LEAF)])
            life = random.uniform(0.35, 0.65)
            self.particles.append([x, y, math.cos(a) * v, math.sin(a) * v - 40, life, life, col,
                                   random.uniform(2, 3.6)])
        self.popups.append([x, y - self.cell * 0.4, "+1", 0.8, 0.8])

    def ping(self, cell_pos):
        """Expanding ring, used when the user drops food by clicking."""
        x, y = self.center(cell_pos)
        self.rings.append([x, y, 0.5, 0.5])

    def render(self, snake, prev_snake, grew, t, food, dead=False):
        now = time.perf_counter()
        dt = min(0.05, now - self._last)
        self._last = now
        self.c.delete("dyn")
        self._draw_hover()
        self._draw_food(food, now)
        pts, length = self._body_samples(snake, prev_snake, grew, t, now, dead)
        head = self._head_frame(pts)
        # The head's shadow and outline go under the body's fill so the neck flows into it.
        self._draw_body(pts, length, dead,
                        after_shadow=lambda: self._head_under(head, food, dead, now, "shadow"),
                        after_outline=lambda: self._head_under(head, food, dead, now, "outline"))
        self._draw_head(head, food, dead)
        self._draw_fx(dt)

    # ---------- ground (drawn once) ----------
    def draw_ground(self, seed):
        """Paint the ground once into a single image (far cheaper than hundreds of items)."""
        C, rng = self.cell, random.Random(seed)
        W = self.grid * C
        a, b = hex_rgb(GROUND_A), hex_rgb(GROUND_B)
        px = [[a if ((x // C) + (y // C)) % 2 else b for x in range(W)] for y in range(W)]
        for _ in range(self.grid * self.grid * 3):              # soil speckles
            cx, cy = rng.randrange(W), rng.randrange(W)
            col = hex_rgb(rng.choice(SPECKLES))
            r = rng.choice((0, 1, 1, 2))
            for yy in range(cy - r, cy + r + 1):
                for xx in range(cx - r, cx + r + 1):
                    if 0 <= xx < W and 0 <= yy < W and (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r + 1:
                        px[yy][xx] = col
        tuft = hex_rgb("#28362a")
        for _ in range(self.grid * 2):                           # little grass tufts
            x0, y0 = rng.randrange(4, W - 4), rng.randrange(8, W - 1)
            for k in (-1, 0, 1):
                h = rng.randint(4, 7)
                for t in range(h):
                    xx = int(round(x0 + k * 2 + k * 1.5 * t / h))
                    if 0 <= xx < W:
                        px[y0 - t][xx] = tuft
        self.ground_img = tk.PhotoImage(width=W, height=W)
        self.ground_img.put(" ".join("{" + " ".join("#%02x%02x%02x" % c for c in row) + "}" for row in px))
        self.c.delete("ground")
        self.c.create_image(0, 0, image=self.ground_img, anchor="nw", tags="ground")

    # ---------- geometry ----------
    def _body_samples(self, snake, prev, grew, t, now, dead):
        """Sample points along the (interpolated) body, head first.
        Each sample: [x, y, dx, dy, dist_from_head, dist_from_tail] in pixels / cells."""
        C = self.cell
        Q = [self.center(p) for p in snake]
        extra = prev is not None and not grew
        if extra:
            Q.append(self.center(prev[-1]))
        if prev is None:
            t = 1.0
        t = smoothstep(t) * 0.35 + t * 0.65      # slight ease, keeps motion lively
        L = len(Q) - 1
        if L == 0:
            x, y = Q[0]
            return [[x, y, 1.0, 0.0, 0.0, 0.0]], 0.0
        s0 = 1.0 - t
        s1 = (L - t) if extra else float(L)
        length = s1 - s0
        pts = []
        s = s0
        while True:
            s = min(s, s1)
            k = min(int(s), L - 1)
            f = s - k
            (ax, ay), (bx, by) = Q[k], Q[k + 1]
            pts.append([ax + (bx - ax) * f, ay + (by - ay) * f,
                        (ax - bx) / C, (ay - by) / C, s - s0, s1 - s])
            if s >= s1:
                break
            r = self._radius(s - s0, length) / C
            s += max(0.06, min(0.2, 0.7 * r))
        n = len(pts)

        # round off the corners with a small moving average (endpoints fixed)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        w = 2
        for i in range(1, n - 1):
            lo, hi = max(0, i - w), min(n, i + w + 1)
            pts[i][0] = sum(xs[lo:hi]) / (hi - lo)
            pts[i][1] = sum(ys[lo:hi]) / (hi - lo)

        # gentle slither: a lateral wave travelling down the body
        if not dead:
            amp = 0.055 * C
            for p in pts:
                d, b = p[4], p[5]
                taper = smoothstep(d / 0.9) * smoothstep(b / 0.6)
                off = amp * taper * math.sin(b * 2 * math.pi / 1.7 + now * 7.0)
                p[0] += -p[3] * off
                p[1] += p[2] * off
        return pts, length

    def _radius(self, d, length):
        C = self.cell
        u = d / length if length > 0 else 0
        taper = 1 - 0.7 * smoothstep((u - 0.3) / 0.7)
        neck = 1 - 0.13 * math.exp(-((d - 0.42) / 0.16) ** 2)
        return max(0.08 * C, BODY_R * C * taper * neck)

    # ---------- body ----------
    def _draw_body(self, pts, length, dead, after_shadow=None, after_outline=None):
        c = self.c
        head_col, tail_col = (DEAD_HEAD, DEAD_TAIL) if dead else (BODY_HEAD, BODY_TAIL)
        body = [p for p in pts if p[4] > 0.14][::-1]          # tail first, head drawn on top
        if len(body) < 2:
            after_shadow and after_shadow()
            after_outline and after_outline()
            return
        geo = []
        for x, y, _, _, d, b in body:
            u = d / length if length > 0 else 0
            geo.append((x, y, self._radius(d, length), mix(head_col, tail_col, u), b))

        # Group consecutive samples into short polylines; each is one canvas item per layer.
        K = 5
        chunks = []
        for i in range(0, len(geo) - 1, K):
            seg = geo[i:i + K + 1]
            r = sum(g[2] for g in seg) / len(seg)
            base = seg[len(seg) // 2][3]
            chunks.append(([v for g in seg for v in (g[0], g[1])], r, base))

        def stroke(coords, width, fill, ox=0.0, oy=0.0):
            if ox or oy:
                coords = [v + (ox if k % 2 == 0 else oy) for k, v in enumerate(coords)]
            c.create_line(coords, width=max(1.0, width), fill=fill, capstyle="round",
                          joinstyle="round", tags="dyn")

        for xy, r, base in chunks:                             # ground shadow
            stroke(xy, 2 * r, SHADOW, 3, 4)
        if after_shadow:
            after_shadow()
        for xy, r, base in chunks:                             # dark outline
            stroke(xy, 2 * r + 2.6, self.col(mix(base, (0, 0, 0), 0.6)))
        if after_outline:
            after_outline()
        for xy, r, base in chunks:                             # body colour
            stroke(xy, 2 * r, self.col(base))
        band = []                                              # saddle markings, one stroke per band
        for x, y, r, base, b in geo + [(0, 0, 0, None, -1.0)]:
            if base is not None and (b / 0.78) % 1.0 < 0.34:
                band.append((x, y, r, base))
                continue
            if band:
                rr = max(g[2] for g in band) * 0.62
                col = self.col(mix(band[len(band) // 2][3], SADDLE, 0.62))
                xy = [v for g in band for v in (g[0], g[1])]
                if len(band) == 1:
                    xy += xy
                stroke(xy, 2 * rr, col)
                band = []
        for xy, r, base in chunks:                             # glossy ridge along the back
            stroke(xy, 0.55 * r, self.col(mix(base, (255, 255, 255), 0.2)), -0.2 * r, -0.3 * r)
        for i, (x, y, r, base, _) in enumerate(geo):           # scale glints
            if i % 5 == 2 and r > 0.12 * self.cell:
                g = r * 0.13
                gx, gy = x - 0.25 * r, y - 0.38 * r
                c.create_oval(gx - g, gy - g, gx + g, gy + g,
                              fill=self.col(mix(base, (255, 255, 255), 0.5)), width=0, tags="dyn")

    # ---------- head ----------
    def _head_frame(self, pts):
        """Head position and a local->screen transform (x forward, y sideways, cell units)."""
        C = self.cell
        hx, hy = pts[0][0], pts[0][1]
        # facing = from a point just behind the head to the head (turns smoothly at corners)
        back = min(pts, key=lambda p: abs(p[4] - 0.35)) if len(pts) > 1 else None
        if back is not None and (abs(hx - back[0]) + abs(hy - back[1])) > 1e-3:
            vx, vy = hx - back[0], hy - back[1]
        else:
            vx, vy = pts[0][2], pts[0][3]
        ang = math.atan2(vy, vx)
        ca, sa = math.cos(ang), math.sin(ang)

        def tr(lx, ly, scale=1.0, ox=0.0, oy=0.0):
            lx, ly = lx * scale, ly * scale
            return hx + (lx * ca - ly * sa) * C + ox, hy + (lx * sa + ly * ca) * C + oy

        return hx, hy, ca, sa, tr

    def _poly(self, tr, shape, fill, scale=1.0, ox=0.0, oy=0.0):
        flat = [v for p in shape for v in tr(*p, scale=scale, ox=ox, oy=oy)]
        self.c.create_polygon(flat, fill=fill, outline="", smooth=True, tags="dyn")

    def _head_base(self, dead):
        return DEAD_HEAD if dead else mix(BODY_HEAD, (255, 255, 255), 0.06)

    def _head_under(self, head, food, dead, now, layer):
        hx, hy, ca, sa, tr = head
        C = self.cell
        base = self._head_base(dead)
        if layer == "shadow":
            self._poly(tr, HEAD_SHAPE, SHADOW, ox=3, oy=4)
            return
        self._poly(tr, HEAD_SHAPE, self.col(mix(base, (0, 0, 0), 0.6)), scale=1.06)
        # tongue sits under the snout
        if dead:
            self._tongue(tr, 0.22, droop=0.06)
        else:
            fxp, fyp = self.center(food)
            near = math.hypot(fxp - hx, fyp - hy) < 1.8 * C
            cyc = (now * (2.6 if near else 1.1) + self.tongue_phase) % 1.0
            if cyc < 0.32:
                self._tongue(tr, 0.34 * math.sin(cyc / 0.32 * math.pi))

    def _draw_head(self, head, food, dead):
        c, C = self.c, self.cell
        hx, hy, ca, sa, tr = head
        base = self._head_base(dead)
        self._poly(tr, HEAD_SHAPE, self.col(base))
        self._poly(tr, HEAD_MARK, self.col(mix(base, SADDLE, 0.55)))
        r = 0.1 * C
        c.create_oval(hx - 0.07 * C - r, hy - 0.1 * C - r, hx - 0.07 * C + r, hy - 0.1 * C + r,
                      fill=self.col(mix(base, (255, 255, 255), 0.22)), width=0, tags="dyn")

        # nostrils
        for side in (-1, 1):
            nx, ny = tr(0.44, 0.055 * side)
            c.create_oval(nx - 1.2, ny - 1.2, nx + 1.2, ny + 1.2, fill="#10241a", width=0, tags="dyn")

        # eyes: yellow iris, slit pupil that looks toward the food
        fx, fy = self.center(food)
        lx, ly = fx - hx, fy - hy
        dist = math.hypot(lx, ly) or 1.0
        lx, ly = lx / dist, ly / dist
        ir = 0.085 * C
        for side in (-1, 1):
            ex, ey = tr(0.17, 0.2 * side)
            c.create_oval(ex - ir - 1, ey - ir - 1, ex + ir + 1, ey + ir + 1,
                          fill="#0d1a12", width=0, tags="dyn")
            if dead:
                c.create_oval(ex - ir, ey - ir, ex + ir, ey + ir, fill="#9aa093", width=0, tags="dyn")
                k = ir * 0.8
                c.create_line(ex - k, ey - k, ex + k, ey + k, fill="#1a1d1a", width=2, tags="dyn")
                c.create_line(ex - k, ey + k, ex + k, ey - k, fill="#1a1d1a", width=2, tags="dyn")
                continue
            c.create_oval(ex - ir, ey - ir, ex + ir, ey + ir, fill=IRIS, width=0, tags="dyn")
            px, py = ex + lx * ir * 0.35, ey + ly * ir * 0.35
            # slit runs across the head (perpendicular to facing)
            sx, sy = -sa, ca
            pw, ph = ir * 0.22, ir * 0.78
            c.create_polygon(px + sx * ph, py + sy * ph, px + ca * pw, py + sa * pw,
                             px - sx * ph, py - sy * ph, px - ca * pw, py - sa * pw,
                             fill="#111111", smooth=True, tags="dyn")
            g = ir * 0.24
            c.create_oval(ex - ir * 0.35 - g, ey - ir * 0.4 - g, ex - ir * 0.35 + g, ey - ir * 0.4 + g,
                          fill="#fffbe6", width=0, tags="dyn")

    def _tongue(self, tr, length, droop=0.0):
        if length <= 0.01:
            return
        x0, y0 = tr(0.46, 0.0)
        x1, y1 = tr(0.48 + length, droop)
        fork = 0.07 * (length / 0.34) + 0.02
        a1 = tr(0.48 + length + fork, droop + fork * 0.75)
        a2 = tr(0.48 + length + fork, droop - fork * 0.75)
        self.c.create_line(x0, y0, x1, y1, fill=TONGUE, width=2, capstyle="round", tags="dyn")
        self.c.create_line(x1, y1, *a1, fill=TONGUE, width=1.6, capstyle="round", tags="dyn")
        self.c.create_line(x1, y1, *a2, fill=TONGUE, width=1.6, capstyle="round", tags="dyn")

    # ---------- food ----------
    def _draw_food(self, food, now):
        c, C = self.c, self.cell
        cx, cy = self.center(food)
        r = 0.26 * C
        bob = math.sin(now * 3.2) * 1.8
        glow = 0.18 + 0.1 * math.sin(now * 3.2)
        g = r * 1.55
        c.create_oval(cx - g, cy - g, cx + g, cy + g, outline=self.col(mix(hex_rgb(GROUND_A), hex_rgb(APPLE), glow)),
                      width=2, tags="dyn")
        sw, sh = r * (0.85 - bob * 0.02), r * 0.24
        c.create_oval(cx - sw, cy + r * 0.95 - sh, cx + sw, cy + r * 0.95 + sh, fill=SHADOW, width=0, tags="dyn")
        y = cy + bob - r * 0.05
        lobes = [(-0.27, 0.0), (0.27, 0.0)]
        rr = r * 0.82
        for ox, oy in lobes:
            x0, y0 = cx + ox * r, y + oy * r
            c.create_oval(x0 - rr - 1.3, y0 - rr - 1.3, x0 + rr + 1.3, y0 + rr + 1.3,
                          fill=APPLE_EDGE, width=0, tags="dyn")
        for ox, oy in lobes:
            x0, y0 = cx + ox * r, y + oy * r
            c.create_oval(x0 - rr, y0 - rr, x0 + rr, y0 + rr, fill=APPLE, width=0, tags="dyn")
        s = r * 0.55
        c.create_oval(cx + 0.28 * r - s, y + 0.3 * r - s, cx + 0.28 * r + s, y + 0.3 * r + s,
                      fill=APPLE_SHADE, width=0, tags="dyn")
        h = r * 0.24
        c.create_oval(cx - 0.42 * r - h, y - 0.3 * r - h, cx - 0.42 * r + h, y - 0.3 * r + h,
                      fill=APPLE_LIGHT, width=0, tags="dyn")
        g2 = r * 0.08
        c.create_oval(cx - 0.5 * r - g2, y - 0.4 * r - g2, cx - 0.5 * r + g2, y - 0.4 * r + g2,
                      fill="#ffe3dc", width=0, tags="dyn")
        c.create_line(cx, y - 0.62 * r, cx + 0.14 * r, y - 1.08 * r, fill=STEM, width=2,
                      capstyle="round", tags="dyn")
        leaf = [(0.1, -0.9), (0.45, -1.25), (0.95, -1.12), (0.55, -0.82)]
        c.create_polygon([v for px, py in leaf for v in (cx + px * r, y + py * r)],
                         fill=LEAF, outline=LEAF_EDGE, smooth=True, tags="dyn")

    # ---------- hover + effects ----------
    def _draw_hover(self):
        if self.hover is None:
            return
        x, y = self.hover
        C = self.cell
        self.c.create_rectangle(x * C + 2, y * C + 2, (x + 1) * C - 2, (y + 1) * C - 2,
                                outline="#3d5240", width=1, dash=(3, 3), tags="dyn")

    def _draw_fx(self, dt):
        c = self.c
        ground = hex_rgb(GROUND_A)
        alive = []
        for p in self.particles:
            p[0] += p[2] * dt
            p[1] += p[3] * dt
            p[3] += 260 * dt
            p[2] *= 0.96
            p[4] -= dt
            if p[4] > 0:
                k = p[4] / p[5]
                r = p[7] * (0.4 + 0.6 * k)
                c.create_oval(p[0] - r, p[1] - r, p[0] + r, p[1] + r,
                              fill=self.col(mix(ground, p[6], k)), width=0, tags="dyn")
                alive.append(p)
        self.particles = alive

        alive = []
        for p in self.popups:
            p[1] -= 34 * dt
            p[3] -= dt
            if p[3] > 0:
                k = p[3] / p[4]
                c.create_text(p[0], p[1], text=p[2], fill=self.col(mix(ground, (255, 230, 120), k)),
                              font=("Segoe UI", 12, "bold"), tags="dyn")
                alive.append(p)
        self.popups = alive

        alive = []
        for p in self.rings:
            p[2] -= dt
            if p[2] > 0:
                k = p[2] / p[3]
                r = self.cell * (0.25 + 0.5 * (1 - k))
                c.create_oval(p[0] - r, p[1] - r, p[0] + r, p[1] + r,
                              outline=self.col(mix(ground, (255, 200, 120), k)), width=2, tags="dyn")
                alive.append(p)
        self.rings = alive
