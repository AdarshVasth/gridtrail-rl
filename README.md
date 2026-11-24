# GridTrail RL 🧭

A small, inspectable **reinforcement-learning** project: a tabular Q-learning agent learns routes through three original mazes by trying actions and receiving rewards. Replay its learned route, compare it with a random policy and a BFS shortest-path reference, and inspect the four action values at the current cell.

Runs locally with NumPy and Streamlit, with no API keys, model downloads, paid services, GPU, or external dataset. This is a deterministic educational simulation, not a real robot controller or a model that transfers to new mazes.

## Run

Tested on Linux, Python 3.12. Other operating systems and Python versions have not been verified.

```bash
git clone https://github.com/AdarshVasth/gridtrail-rl.git
cd gridtrail-rl
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\Activate.ps1` in Windows PowerShell, then:

```bash
python -m pip install -r requirements-tested.txt
python -m streamlit run app.py --server.address 127.0.0.1
```

Open the local URL printed by Streamlit. Choose a maze, training seed and episode budget. The app trains from zero for each configuration (cached to avoid repeating identical work), then exposes a replay slider and the learned Q values. Switch to a random-policy route to see the difference. The Q table can be downloaded as JSON; no arbitrary model files are loaded.

`requirements-tested.txt` pins the exact verified environment. `requirements.txt` provides version ranges for a fresh resolver. Initial dependency installation needs internet; subsequent training and inference need none. Streamlit telemetry is disabled in project configuration. There are no external calls in application code.

## What is actually learned?

The Q table starts at zero: one row per grid cell and four columns for up/right/down/left. Each step updates only the action just taken:

`Q(s,a) ← Q(s,a) + 0.25 × [reward + 0.95 × max(Q(next_state)) − Q(s,a)]`

For terminal goal transitions the target is just `reward`, with no bootstrap. Exploration starts at epsilon=1 and decays as `max(0.05, 0.995**episode)`; ties are broken randomly during training. Greedy evaluation uses deterministic first-maximum tie-breaking. Ordinary moves cost −1, blocked moves cost −2, reaching the goal pays +20, and the goal is absorbing. Episodes reset to the fixed start and are capped at 128 actions. The cap is a truncation, not a terminal state, so its last transition still bootstraps.

The learner gets sampled transitions through `Grid.step`; it does **not** use BFS, shortest-path labels, or a precomputed route. BFS is an independent known-map reference used only after training. For a fully known static map, BFS is simpler and more efficient than learning; RL here teaches learning from interaction.

## Measured results

`python evaluate.py` actually trained 9 agents: 3 maps × seeds 7, 19, 42, each for 2,500 episodes. All nine greedy routes reached their goal and matched the BFS path length on their own training map:

| Original map | Learned actions, all 3 seeds | BFS shortest | Random success within 128 actions |
|---|---:|---:|---:|
| courtyard | 14 | 14 | 10/100 |
| switchback | 28 | 28 | 0/100 |
| crossroads | 14 | 14 | 16/100 |

The random baseline uses fixed seeds 10000–10099 and is the same for each training seed. Its mean capped actions count failures as 128; it is not the average path length conditional on success. A successful switchback route has negative **undiscounted** total reward (−7); the displayed success flag and shortest path are the clearer checks. Training optimizes discounted return, while the displayed episode total is undiscounted.

These are **training-environment solution checks**, not held-out accuracy or evidence of generalization. The greedy route is deterministic from one start; repeating it 100 times would not provide 100 independent tests. Results are recorded in `data/benchmark.json`, not manually invented. Other seeds or smaller budgets may fail or learn looping policies.

## Verify

```bash
python evaluate.py
python -m pytest -q
python -m pip check
```

The evaluation script overwrites only `data/benchmark.json`. Tests exercise environment transitions, invalid maps/actions, terminal TD updates, reproducibility, actual learning from a zero table, independence from BFS, shortest-path agreement, unreachable goals/timeouts, renderer shape and Streamlit control interactions. Exact floating-point Q values can vary across library versions; compare behavior as well as values.

## Architecture

- `agent.py`: validated grid environment, Q-learning, rollouts, independent BFS reference, RGB replay rendering.
- `app.py`: interactive Streamlit demo, bounded training cache, Q-value table, training curve and JSON export.
- `evaluate.py`: fixed multi-seed experiment and random-policy baseline.
- `data/maps.json`: three original hand-authored 8×8 mazes; no third-party map assets.
- `tests/`: numerical, behavioral, separation-of-planner and UI tests.
- `docs/LEARNING.md`: short exercises and an internship explanation.
- `docs/VALIDATION.md`: what was actually verified and what was not.

## Limitations

The state is a cell index tied to a fixed layout and goal. Each map requires retraining; a Q table of the same shape is not interchangeable between maps. The API validates shape and finiteness but does not encode map identity in a model object. There are no moving obstacles, partial observations, sensor noise, continuous actions or safety constraints. No convergence guarantee is claimed: learning uses finite episodes, fixed learning rate and fixed starts, which do not establish the assumptions of asymptotic Q-learning convergence theorems. UI samples are fixed; adding custom mazes requires editing the JSON and checking reachability. There is no public deployment or authentication.

## Sources and rights

Current APIs checked on 2026-11-24 against [NumPy Generator](https://numpy.org/doc/stable/reference/random/generator.html), [Streamlit image](https://docs.streamlit.io/develop/api-reference/media/st.image), and [Streamlit AppTest](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest) official documentation. Algorithm background: [Watkins & Dayan, Q-learning (1992)](https://www.gatsby.ucl.ac.uk/~dayan/papers/wd92.html). The implementation and mazes are original project code/data, not copied paper code or copyrighted game maps. All included code and sample maps are redistributable under MIT; no credentials or personal records are included.
