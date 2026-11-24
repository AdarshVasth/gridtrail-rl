"""Train actual agents and save reproducible evaluation artifacts."""
import json
from pathlib import Path
from agent import load_maps, train, evaluate, rollout

if __name__ == '__main__':
    result = {}
    for name, grid in load_maps().items():
        runs = []
        for seed in [7, 19, 42]:
            q, _ = train(grid, seed=seed)
            runs.append({'seed': seed, **evaluate(grid, q)})
        result[name] = runs
    target = Path(__file__).parent / 'data/benchmark.json'
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
