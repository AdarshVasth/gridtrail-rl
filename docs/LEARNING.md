# Learn GridTrail in about an hour

1. **Environment, 10 minutes.** Read `Grid.step`. A state is a cell index; an action is one of four directions. Draw the transition from a wall-adjacent cell. A blocked action stays in place and receives −2. A goal receives +20 once, then becomes absorbing.
2. **Update, 10 minutes.** With old Q=0, next best Q=8, reward=−1, rate=.25 and discount=.95, the new value is `.25 * (-1 + .95*8) = 1.65`. For a terminal reward 20 it becomes 5 regardless of next Q. Find the test proving this.
3. **Exploration, 10 minutes.** Why does always choosing the first best action from an all-zero table get stuck? Compare epsilon-greedy exploration and deterministic greedy evaluation. A seed controls random choices; it does not prove a universal result.
4. **Run the demo, 10 minutes.** Set 100 episodes, then 2500. Inspect the route and Q values. At a goal the four values remain zero because the agent never takes an action there. Wall rows also stay unused.
5. **Evaluation, 10 minutes.** BFS knows the full map; Q-learning only samples transitions. Inspect `test_training_does_not_use_bfs`, which replaces the planner with a function that fails if called. Explain why nine solved training maps do not measure transfer to unseen layouts.
6. **Read the cap, 10 minutes.** A 128-step limit stops the rollout, but does not make a non-goal cell terminal in the MDP. Explain why the last training transition still bootstraps. Why can a correct 28-step path have total reward −7?

## Experiments

- Add a fourth original map with a reachable goal. Verify reachability with BFS but never feed the path into training.
- Measure time-to-first-success across ten new seeds, reporting failed seeds too.
- Change the blocked-action penalty. Record how exploration cost changes before evaluating the greedy route.
- Introduce random action slips; redesign evaluation as multiple independent episodes and report success uncertainty. Do not reuse deterministic one-route scores.
- Try random starting cells and evaluate only reachable non-wall cells. This tests coverage of the state table, not transfer to a different maze.

## Interview explanation

“I implemented tabular Q-learning from scratch with NumPy, including epsilon-greedy exploration and terminal targets. I built a visual replay so I could inspect failures and Q values. I checked learned routes against an independent BFS reference and tested that the learner never calls that planner. It solves these tiny training environments, but it has no generalization across layouts and does not establish a convergence guarantee.”

Know the differences between a value function and a policy, exploration and exploitation, model-free learning and known-map planning, discounted return and displayed total reward, termination and truncation, and reproducibility and generalization.
