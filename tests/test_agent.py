import numpy as np
import pytest
from agent import Grid, load_maps, train, td_update, rollout, shortest_path, render_grid


def test_wall_boundary_and_goal_absorbing():
    g = Grid(('S#', '.G'))
    assert g.step(g.start, 0) == (g.start, -2., False)
    assert g.step(g.start, 1) == (g.start, -2., False)
    assert g.step(2, 1) == (g.goal, 20., True)
    assert g.step(g.goal, 0) == (g.goal, 0., True)


@pytest.mark.parametrize('rows', [(), ('S', 'G'), ('S..', '.G'), ('SS', '.G'), ('Sx', '.G')])
def test_invalid_maps(rows):
    with pytest.raises(ValueError):
        Grid(rows)


def test_bfs_known_solution_and_unreachable():
    assert shortest_path(Grid(('S.', '.G'))) == [0, 1, 3]
    assert shortest_path(Grid(('S#', '#G'))) == []


def test_terminal_update_does_not_bootstrap():
    q = np.zeros((3, 4))
    q[2] = 100
    td_update(q, 0, 1, 20, 2, True, rate=.5)
    assert q[0, 1] == 10
    td_update(q, 1, 2, -1, 2, False, rate=.5, discount=.9)
    assert q[1, 2] == 44.5


def test_reproducible_real_learning_and_untrained_baseline():
    g = load_maps()['courtyard']
    q, history = train(g, seed=7)
    again, repeated = train(g, seed=7)
    np.testing.assert_array_equal(q, again)
    assert history == repeated and len(history) == 2500
    assert not np.all(q == 0)
    assert not rollout(g, np.zeros_like(q))['success']
    assert rollout(g, q)['success']


@pytest.mark.parametrize('name', ['courtyard', 'switchback', 'crossroads'])
def test_actual_training_reaches_goal_and_matches_shortest_path(name):
    g = load_maps()[name]
    q, _ = train(g, seed=19)
    run = rollout(g, q)
    assert run['success']
    assert run['steps'] == len(shortest_path(g)) - 1
    for state in run['path']:
        g.check_state(state)


def test_training_does_not_use_bfs(monkeypatch):
    import agent
    def forbidden(*args):
        raise AssertionError('Planner leaked into learner')
    monkeypatch.setattr(agent, 'shortest_path', forbidden)
    q, _ = train(Grid(('S.', '.G')), episodes=100)
    assert rollout(Grid(('S.', '.G')), q)['success']


def test_timeout_and_input_rejection():
    g = Grid(('S#', '#G'))
    assert rollout(g, max_steps=3)['steps'] == 3
    assert not rollout(g, max_steps=3)['success']
    for q in [np.zeros((1,4)), np.full((g.size,4), np.nan)]:
        with pytest.raises(ValueError):
            rollout(g, q)
    with pytest.raises(ValueError):
        train(g, episodes=0)
    with pytest.raises(ValueError):
        g.step(1, 0)
    with pytest.raises(ValueError):
        g.step(0, -1)


def test_render_is_rgb_and_does_not_mutate_map():
    g = Grid(('S.', '.G'))
    img = render_grid(g, [0,1,3], 1)
    assert img.shape == (96,96,3) and img.dtype == np.uint8
    assert g.rows == ('S.', '.G')
    with pytest.raises(ValueError):
        render_grid(g, [0], 3)
