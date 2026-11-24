"""Finite deterministic gridworld, tabular Q-learning and independent BFS reference."""
from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np

ACTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
ACTION_NAMES = ('up', 'right', 'down', 'left')


@dataclass(frozen=True)
class Grid:
    rows: tuple[str, ...]

    def __post_init__(self):
        if not self.rows or not 2 <= len(self.rows) <= 20:
            raise ValueError('Grid height must be 2–20.')
        if not 2 <= len(self.rows[0]) <= 20 or any(len(r) != len(self.rows[0]) for r in self.rows):
            raise ValueError('Use a rectangular grid of width 2–20.')
        chars = ''.join(self.rows)
        if set(chars) - set('.#SG') or chars.count('S') != 1 or chars.count('G') != 1:
            raise ValueError('Use . # and exactly one S and G.')

    @property
    def width(self):
        return len(self.rows[0])

    @property
    def size(self):
        return len(self.rows) * self.width

    def locate(self, symbol):
        return ''.join(self.rows).index(symbol)

    @property
    def start(self):
        return self.locate('S')

    @property
    def goal(self):
        return self.locate('G')

    def check_state(self, state):
        if not isinstance(state, (int, np.integer)) or not 0 <= state < self.size:
            raise ValueError('Invalid state.')
        if self.rows[state // self.width][state % self.width] == '#':
            raise ValueError('State is a wall.')

    def step(self, state, action):
        self.check_state(state)
        if not isinstance(action, (int, np.integer)) or not 0 <= action < 4:
            raise ValueError('Action must be an integer from 0 to 3.')
        if state == self.goal:
            return state, 0.0, True
        r, c = divmod(int(state), self.width)
        dr, dc = ACTIONS[action]
        nr, nc = r + dr, c + dc
        if not 0 <= nr < len(self.rows) or not 0 <= nc < self.width or self.rows[nr][nc] == '#':
            return int(state), -2.0, False
        new = nr * self.width + nc
        return new, 20.0 if new == self.goal else -1.0, new == self.goal


def load_maps():
    raw = json.loads((Path(__file__).parent / 'data/maps.json').read_text())
    return {name: Grid(tuple(rows)) for name, rows in raw.items()}


def shortest_path(grid):
    """Known-map BFS for evaluation only. Never called by training."""
    queue = deque([(grid.start, [grid.start])])
    seen = {grid.start}
    while queue:
        state, path = queue.popleft()
        if state == grid.goal:
            return path
        for action in range(4):
            new, _, _ = grid.step(state, action)
            if new not in seen:
                seen.add(new)
                queue.append((new, path + [new]))
    return []


def td_update(q, state, action, reward, next_state, terminal, rate=.25, discount=.95):
    """One Q-learning update; terminal transitions have no bootstrap term."""
    target = reward if terminal else reward + discount * float(q[next_state].max())
    q[state, action] += rate * (target - q[state, action])


def train(grid, seed=7, episodes=2500, max_steps=128):
    if not isinstance(episodes, int) or not 1 <= episodes <= 10000:
        raise ValueError('Episodes must be 1–10000.')
    if not isinstance(max_steps, int) or not 1 <= max_steps <= 2000:
        raise ValueError('max_steps must be 1–2000.')
    rng = np.random.default_rng(seed)
    q = np.zeros((grid.size, 4), dtype=float)
    history = []
    for episode in range(episodes):
        epsilon = max(.05, .995 ** episode)
        state, total, success = grid.start, 0., False
        for step in range(max_steps):
            if rng.random() < epsilon:
                action = int(rng.integers(4))
            else:
                # Random tie-breaking during learning avoids a fixed directional bias.
                action = int(rng.choice(np.flatnonzero(q[state] == q[state].max())))
            new, reward, terminal = grid.step(state, action)
            td_update(q, state, action, reward, new, terminal)
            total += reward
            state = new
            if terminal:
                success = True
                break
        history.append(dict(episode=episode + 1, reward=total, steps=step + 1, success=success))
    return q, history


def rollout(grid, q=None, seed=2026, max_steps=128):
    if not isinstance(max_steps, int) or not 1 <= max_steps <= 2000:
        raise ValueError('max_steps must be 1–2000.')
    if q is not None:
        q = np.asarray(q, dtype=float)
        if q.shape != (grid.size, 4) or not np.isfinite(q).all():
            raise ValueError('Q table must have one finite row per grid cell and four actions.')
    rng = np.random.default_rng(seed)
    state, path, total = grid.start, [grid.start], 0.
    for _ in range(max_steps):
        action = int(rng.integers(4)) if q is None else int(np.argmax(q[state]))
        state, reward, done = grid.step(state, action)
        path.append(state)
        total += reward
        if done:
            return dict(path=path, success=True, reward=total, steps=len(path) - 1)
    return dict(path=path, success=False, reward=total, steps=max_steps)


def evaluate(grid, q, trials=100):
    if not isinstance(trials, int) or not 1 <= trials <= 1000:
        raise ValueError('Trials must be 1–1000.')
    learned = rollout(grid, q)
    random_runs = [rollout(grid, seed=10000 + i) for i in range(trials)]
    reference = shortest_path(grid)
    return {'greedy_success': learned['success'], 'greedy_steps': learned['steps'],
        'greedy_reward': learned['reward'], 'optimal_steps': len(reference)-1 if reference else None,
        'random_trials': trials, 'random_success_rate': sum(x['success'] for x in random_runs) / trials,
        'random_mean_capped_steps': float(np.mean([x['steps'] for x in random_runs])),
        'max_steps': 128}


def render_grid(grid, path, step):
    if not path or not isinstance(step, int) or not 0 <= step < len(path):
        raise ValueError('Select a valid path step.')
    for state in path:
        grid.check_state(state)
    colors = {'.': (237, 242, 247), '#': (35, 47, 63), 'S': (68, 145, 209), 'G': (42, 170, 119)}
    pixels = np.array([[colors[ch] for ch in row] for row in grid.rows], dtype=np.uint8)
    for state in path[:step + 1]:
        if state not in (grid.start, grid.goal):
            pixels[divmod(state, grid.width)] = (249, 213, 107)
    pixels[divmod(path[step], grid.width)] = (222, 83, 86)
    image = np.repeat(np.repeat(pixels, 48, axis=0), 48, axis=1)
    image[::48, :, :] = 255
    image[:, ::48, :] = 255
    return image
