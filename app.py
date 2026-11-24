import json
import streamlit as st
from agent import load_maps, train, rollout, evaluate, render_grid, ACTION_NAMES

st.set_page_config(page_title='GridTrail RL', page_icon='🧭', layout='wide')
st.title('🧭 GridTrail RL')
st.caption('Learn a maze by trial and reward — inspect what the agent learned')
st.info('An educational, deterministic simulation. Each map gets its own Q table; this agent does not generalize to new layouts or control a real robot.')
maps = load_maps()
name = st.selectbox('Maze', list(maps))
seed = int(st.number_input('Training seed', min_value=0, max_value=99999, value=7))
episodes = st.select_slider('Training episodes', options=[100, 500, 1000, 2500], value=2500)
grid = maps[name]

@st.cache_data(max_entries=12)
def learn(name, seed, episodes):
    q, history = train(maps[name], seed=seed, episodes=episodes)
    return q, history, evaluate(maps[name], q)

with st.spinner('Training the Q table locally…'):
    q, history, metrics = learn(name, seed, episodes)
mode = st.radio('Route to inspect', ['Learned greedy policy', 'Random policy'])
result = rollout(grid, q if mode == 'Learned greedy policy' else None, seed=2026)
if not result['success']:
    st.warning('The route did not reach the goal within 128 actions. Try more training episodes or inspect where it loops.')
a, b, c = st.columns(3)
a.metric('Reached goal', 'Yes' if result['success'] else 'No')
b.metric('Actions taken (capped)', result['steps'])
c.metric('BFS shortest route', metrics['optimal_steps'])
step = st.slider('Replay step', 0, result['steps'], 0, key=f'replay_{name}_{mode}_{episodes}_{seed}')
st.image(render_grid(grid, result['path'], step), width=420, caption='Blue: start · green: goal · gold: visited · red: current position · dark: wall')
state = result['path'][step]
st.write(f'Current cell: row {state // grid.width}, column {state % grid.width} (zero-based)')
st.dataframe([{'action': action, 'learned_Q': float(q[state, i])} for i, action in enumerate(ACTION_NAMES)], hide_index=True)
with st.expander('Training curve and baseline', expanded=False):
    # Display block means rather than 2500 individual episode traces.
    blocks = [history[i:i + 100] for i in range(0, len(history), 100)]
    st.line_chart([sum(r['reward'] for r in block) / len(block) for block in blocks], x_label='100-episode block (zero-based)', y_label='Mean reward')
    st.write(f"Random-policy success: {metrics['random_success_rate']:.0%} over {metrics['random_trials']} fixed-seed trials, each capped at 128 actions.")
    st.write('The learned-policy rollout is deterministic from one fixed start. These are training-map checks, not held-out generalization scores. BFS knows the whole map; it is a reference, not part of learning.')
st.download_button('Download learned Q table (JSON)', json.dumps(q.tolist(), indent=2), f'{name}_q.json', 'application/json')
st.markdown('**Learning rule:** update one state/action value toward `reward + 0.95 × max(next Q)`. Terminal goals use reward only. Each ordinary move costs −1; a blocked move costs −2; reaching the goal pays +20. See `docs/LEARNING.md` for exercises.')
