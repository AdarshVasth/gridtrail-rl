# Verification record — 2026-11-24

Tested locally on Linux, Python 3.12, NumPy 2.5.5, Streamlit 1.67.0 and pytest 9.2.0. Full exact dependency environment is recorded in `requirements-tested.txt`.

- Actual Q-learning training executed for 3 maps × 3 seeds (7, 19, 42), 2500 episodes per run. All 9 greedy routes reached the goal and matched BFS path length: courtyard 14, switchback 28, crossroads 14. These are training-map checks, not held-out generalization.
- Actual random baseline: 100 fixed seeds per map, 128-action cap; success 10%, 0%, 16% respectively. Repeated baseline entries in benchmark JSON are the same runs, not independent repetitions.
- `python -m pytest -q`: **16 passed**. Includes actual model learning, deterministic reproducibility, untrained-policy failure, terminal/no-bootstrap arithmetic, wall/boundary handling, unreachable goal cap, invalid map/action/Q inputs, planner independence, optimal-path agreement and Streamlit AppTest route/map/replay interactions.
- `python -m pip check`: **No broken requirements found**.
- `python -m compileall -q agent.py app.py evaluate.py tests`: passed.
- An actual Streamlit subprocess was launched on loopback and `/_stcore/health` returned **200 ok**; the process was then stopped. This is a liveness check. AppTest separately executed UI application logic and control changes.

No public deployment, real-robot test, unseen-maze generalization test, Windows/macOS run, or manual browser visual audit was performed. Download behavior was not browser-tested. No convergence guarantee or production-readiness claim is made.
