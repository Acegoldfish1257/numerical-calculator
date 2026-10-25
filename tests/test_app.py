from pathlib import Path
from streamlit.testing.v1 import AppTest
import pytest
from services.catalog import METHODS

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_bisection_page():
    app = AppTest.from_file(APP, default_timeout=30).run()
    assert not app.exception
    app.button[0].click().run()
    assert not app.exception and not app.error
    assert app.metric[0].value.startswith("1.521")
    assert len(app.dataframe[0].value) > 0
    app.text_input(key="Bisection:f").set_value("__import__('os')")
    app.button[0].click().run()
    assert app.error and not app.exception


@pytest.mark.parametrize("method", METHODS, ids=lambda m: m.name)
def test_default_page(method):
    app = AppTest.from_file(APP, default_timeout=30).run()
    app.selectbox(key="topic").select(method.topic).run()
    app.selectbox(key="method").select(method.name).run()
    app.button[0].click().run()
    assert not app.exception
    assert not app.error
    assert len(app.dataframe) > 0


@pytest.mark.parametrize("method", METHODS, ids=lambda m: m.name)
def test_bad_input_never_crashes(method):
    app = AppTest.from_file(APP, default_timeout=30).run()
    app.selectbox(key="topic").select(method.topic).run()
    app.selectbox(key="method").select(method.name).run()
    field = next(field for field in method.fields if field.kind in ("text", "area"))
    widget = app.text_area if field.kind == "area" else app.text_input
    widget(key=f"{method.name}:{field.key}").set_value("not valid input!")
    app.button[0].click().run()
    assert not app.exception
    assert app.error
