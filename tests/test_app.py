"""End-to-end Smoke-Test via Streamlits offizielles AppTest-Framework."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


def test_app_loads_without_exception():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=180)
    assert not at.exception, [str(e) for e in at.exception]


def test_preset_buttons_do_not_raise():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=180)
    for button in at.button:
        button.click().run(timeout=180)
        assert not at.exception, [str(e) for e in at.exception]
