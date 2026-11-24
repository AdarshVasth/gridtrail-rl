from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_maze_controls_and_replay():
    app = AppTest.from_file(str(Path(__file__).parents[1] / 'app.py')).run(timeout=45)
    assert not app.exception
    assert app.metric[0].value == 'Yes'
    app.slider[0].set_value(2).run()
    assert not app.exception
    app.selectbox[0].select('switchback').run(timeout=45)
    assert not app.exception and app.metric[0].value == 'Yes'
    app.radio[0].set_value('Random policy').run()
    assert not app.exception
    assert len(app.dataframe) == 1
